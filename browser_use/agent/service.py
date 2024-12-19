from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Type, TypeVar
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	BaseMessage,
	SystemMessage,
 	HumanMessage,
)
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.prompts import AgentMessagePrompt, SystemPrompt
from browser_use.agent.views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentStepInfo,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	AgentEndTelemetryEvent,
	AgentRunTelemetryEvent,
	AgentStepErrorTelemetryEvent,
)
from browser_use.utils import time_execution_async

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class Agent:
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		controller: Controller = Controller(),
		use_vision: bool = True,
		save_conversation_path: Optional[str] = None,
		max_failures: int = 5,
		retry_delay: int = 10,
		system_prompt_class: Type[SystemPrompt] = SystemPrompt,
		max_input_tokens: int = 128000,
		validate_output: bool = False,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'tabindex',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
		],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
	):
		self.agent_id = str(uuid.uuid4())  # unique identifier for the agent

		self.task = task
		self.use_vision = use_vision
		self.llm = llm
		self.save_conversation_path = save_conversation_path
		self._last_result = None
		self.include_attributes = include_attributes
		self.max_error_length = max_error_length
		# Controller setup
		self.controller = controller
		self.max_actions_per_step = max_actions_per_step

		# Browser setup
		self.injected_browser = browser is not None
		self.injected_browser_context = browser_context is not None

		# Initialize browser first if needed
		self.browser = browser if browser is not None else (None if browser_context else Browser())

		# Initialize browser context
		if browser_context:
			self.browser_context = browser_context
		elif self.browser:
			self.browser_context = BrowserContext(
				browser=self.browser, config=self.browser.config.new_context_config
			)
		else:
			# If neither is provided, create both new
			self.browser = Browser()
			self.browser_context = BrowserContext(browser=self.browser)

		self.system_prompt_class = system_prompt_class

		# Telemetry setup
		self.telemetry = ProductTelemetry()

		# Action and output models setup
		self._setup_action_models()

		self.max_input_tokens = max_input_tokens

		self.message_manager = MessageManager(
			llm=self.llm,
			task=self.task,
			action_descriptions=self.controller.registry.get_prompt_description(),
			system_prompt_class=self.system_prompt_class,
			max_input_tokens=self.max_input_tokens,
			include_attributes=self.include_attributes,
			max_error_length=self.max_error_length,
			max_actions_per_step=self.max_actions_per_step,
		)

		# Tracking variables
		self.history: AgentHistoryList = AgentHistoryList(history=[])
		self.n_steps = 1
		self.consecutive_failures = 0
		self.max_failures = max_failures
		self.retry_delay = retry_delay
		self.validate_output = validate_output

		if save_conversation_path:
			logger.info(f'Saving conversation to {save_conversation_path}')

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry"""
		# Get the dynamic action model from controller's registry
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

	@time_execution_async('--step')
	async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
		"""Execute one step of the task"""
		logger.info(f'\n📍 Step {self.n_steps}')
		state = None
		model_output = None
		result: list[ActionResult] = []

		try:
			state = await self.browser_context.get_state(use_vision=self.use_vision)
			self.message_manager.add_state_message(state, self._last_result, step_info)
			input_messages = self.message_manager.get_messages()
			model_output = await self.get_next_action(input_messages)
			self._save_conversation(input_messages, model_output)
			self.message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
			self.message_manager.add_model_output(model_output)

			result: list[ActionResult] = await self.controller.multi_act(
				model_output.action, self.browser_context
			)
			self._last_result = result

			if len(result) > 0 and result[-1].is_done:
				logger.info(f'📄 Result: {result[-1].extracted_content}')

			self.consecutive_failures = 0

		except Exception as e:
			result = self._handle_step_error(e)
			self._last_result = result

		finally:
			if not result:
				return
			for r in result:
				if r.error:
					self.telemetry.capture(
						AgentStepErrorTelemetryEvent(
							agent_id=self.agent_id,
							error=r.error,
						)
					)
			if state:
				self._make_history_item(model_output, state, result)

	def _handle_step_error(self, error: Exception) -> list[ActionResult]:
		"""Handle all types of errors that can occur during a step"""
		include_trace = logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

		if isinstance(error, (ValidationError, ValueError)):
			logger.error(f'{prefix}{error_msg}')
			if 'Max token limit reached' in error_msg:
				# cut tokens from history
				self.message_manager.max_input_tokens = self.max_input_tokens - 500
				logger.info(
					f'Cutting tokens from history - new max input tokens: {self.message_manager.max_input_tokens}'
				)
				self.message_manager.cut_messages()
			self.consecutive_failures += 1
		elif isinstance(error, RateLimitError):
			logger.warning(f'{prefix}{error_msg}')
			time.sleep(self.retry_delay)
			self.consecutive_failures += 1
		else:
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		return [ActionResult(error=error_msg, include_in_memory=True)]

	def _make_history_item(
		self,
		model_output: AgentOutput | None,
		state: BrowserState,
		result: list[ActionResult],
	) -> None:
		"""Create and store history item"""
		interacted_element = None
		len_result = len(result)

		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(
				model_output, state.selector_map
			)
		else:
			interacted_elements = [None]

		state_history = BrowserStateHistory(
			url=state.url,
			title=state.title,
			tabs=state.tabs,
			interacted_element=interacted_elements,
			screenshot=state.screenshot,
		)

		history_item = AgentHistory(model_output=model_output, result=result, state=state_history)

		self.history.history.append(history_item)

	@time_execution_async('--get_next_action')
	async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
		"""Get next action from LLM based on current state"""

		structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
		input_messages_validate_task = input_messages + [HumanMessage(content=self.system_prompt_class.validate_ultimate_task(task=self.task))]
		response: dict[str, Any] = await structured_llm.ainvoke(input_messages_validate_task)  # type: ignore

		parsed: AgentOutput = response['parsed']
		# cut the number of actions to max_actions_per_step
		parsed.action = parsed.action[: self.max_actions_per_step]
		self._log_response(parsed)
		self.n_steps += 1

		return parsed

	def _log_response(self, response: AgentOutput) -> None:
		"""Log the model's response"""
		if 'Success' in response.current_state.evaluation_previous_goal:
			emoji = '👍'
		elif 'Failed' in response.current_state.evaluation_previous_goal:
			emoji = '⚠'
		else:
			emoji = '🤷'

		logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
		logger.info(f'🧠 Memory: {response.current_state.memory}')
		logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
		for i, action in enumerate(response.action):
			logger.info(
				f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}'
			)

	def _save_conversation(self, input_messages: list[BaseMessage], response: Any) -> None:
		"""Save conversation history to file if path is specified"""
		if not self.save_conversation_path:
			return

		# create folders if not exists
		os.makedirs(os.path.dirname(self.save_conversation_path), exist_ok=True)

		with open(self.save_conversation_path + f'_{self.n_steps}.txt', 'w') as f:
			self._write_messages_to_file(f, input_messages)
			self._write_response_to_file(f, response)

	def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
		"""Write messages to conversation file"""
		for message in messages:
			f.write(f' {message.__class__.__name__} \n')

			if isinstance(message.content, list):
				for item in message.content:
					if isinstance(item, dict) and item.get('type') == 'text':
						f.write(item['text'].strip() + '\n')
			elif isinstance(message.content, str):
				try:
					content = json.loads(message.content)
					f.write(json.dumps(content, indent=2) + '\n')
				except json.JSONDecodeError:
					f.write(message.content.strip() + '\n')

			f.write('\n')

	def _write_response_to_file(self, f: Any, response: Any) -> None:
		"""Write model response to conversation file"""
		f.write(' RESPONSE\n')
		f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

	async def run(self, max_steps: int = 100) -> AgentHistoryList:
		"""Execute the task with maximum number of steps"""
		try:
			logger.info(f'🚀 Starting task: {self.task}')

			self.telemetry.capture(
				AgentRunTelemetryEvent(
					agent_id=self.agent_id,
					task=self.task,
				)
			)

			for step in range(max_steps):
				if self._too_many_failures():
					break

				await self.step()

				if self.history.is_done():
					if (
						self.validate_output and step < max_steps - 1
					):  # if last step, we dont need to validate
						if not await self._validate_output():
							continue

					logger.info('✅ Task completed successfully')
					break
			else:
				logger.info('❌ Failed to complete task in maximum steps')

			return self.history

		finally:
			self.telemetry.capture(
				AgentEndTelemetryEvent(
					agent_id=self.agent_id,
					task=self.task,
					success=self.history.is_done(),
					steps=len(self.history.history),
				)
			)
			if not self.injected_browser_context:
				await self.browser_context.close()

			if not self.injected_browser and self.browser:
				await self.browser.close()

	def _too_many_failures(self) -> bool:
		"""Check if we should stop due to too many failures"""
		if self.consecutive_failures >= self.max_failures:
			logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
			return True
		return False

	async def _validate_output(self) -> bool:
		"""Validate the output of the last action is what the user wanted"""
		system_msg = (
			f'You are a validator of an agent who interacts with a browser. '
			f'Validate if the output of last action is what the user wanted and if the task is completed. '
			f'If the task is unclear defined, you can let it pass. But if something is missing or the image does not show what was requested dont let it pass. '
			f'Try to understand the page and help the model with suggestions like scroll, do x, ... to get the solution right. '
			f'Task to validate: {self.task}. Return a JSON object with 2 keys: is_valid and reason. '
			f'is_valid is a boolean that indicates if the output is correct. '
			f'reason is a string that explains why it is valid or not.'
			f' example: {{"is_valid": false, "reason": "The user wanted to search for "cat photos", but the agent searched for "dog photos" instead."}}'
		)

		if self.browser_context.session:
			state = await self.browser_context.get_state(use_vision=self.use_vision)
			content = AgentMessagePrompt(
				state=state,
				result=self._last_result,
				include_attributes=self.include_attributes,
				max_error_length=self.max_error_length,
			)
			msg = [SystemMessage(content=system_msg), content.get_user_message()]
		else:
			# if no browser session, we can't validate the output
			return True

		class ValidationResult(BaseModel):
			is_valid: bool
			reason: str

		validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
		response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
		parsed: ValidationResult = response['parsed']
		is_valid = parsed.is_valid
		if not is_valid:
			logger.info(f'❌ Validator decision: {parsed.reason}')
			msg = f'The ouput is not yet correct. {parsed.reason}.'
			self._last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
		else:
			logger.info(f'✅ Validator decision: {parsed.reason}')
		return is_valid

	async def rerun_history(
		self,
		history: AgentHistoryList,
		max_retries: int = 3,
		skip_failures: bool = True,
		delay_between_actions: float = 2.0,
	) -> list[ActionResult]:
		"""
		Rerun a saved history of actions with error handling and retry logic.

		Args:
		        history: The history to replay
		        max_retries: Maximum number of retries per action
		        skip_failures: Whether to skip failed actions or stop execution
		        delay_between_actions: Delay between actions in seconds

		Returns:
		        List of action results
		"""
		results = []

		for i, history_item in enumerate(history.history):
			goal = (
				history_item.model_output.current_state.next_goal
				if history_item.model_output
				else ''
			)
			logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

			if (
				not history_item.model_output
				or not history_item.model_output.action
				or history_item.model_output.action == [None]
			):
				logger.warning(f'Step {i + 1}: No action to replay, skipping')
				results.append(ActionResult(error='No action to replay'))
				continue

			retry_count = 0
			while retry_count < max_retries:
				try:
					result = await self._execute_history_step(history_item, delay_between_actions)
					results.extend(result)
					break

				except Exception as e:
					retry_count += 1
					if retry_count == max_retries:
						error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
						logger.error(error_msg)
						if not skip_failures:
							results.append(ActionResult(error=error_msg))
							raise RuntimeError(error_msg)
					else:
						logger.warning(
							f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...'
						)
						await asyncio.sleep(delay_between_actions)

		return results

	async def _execute_history_step(
		self, history_item: AgentHistory, delay: float
	) -> list[ActionResult]:
		"""Execute a single step from history with element validation"""

		state = await self.browser_context.get_state()
		if not state or not history_item.model_output:
			raise ValueError('Invalid state or model output')
		updated_actions = []
		for i, action in enumerate(history_item.model_output.action):
			updated_action = await self._update_action_indices(
				history_item.state.interacted_element[i],
				action,
				state,
			)
			updated_actions.append(updated_action)

			if updated_action is None:
				raise ValueError(f'Could not find matching element {i} in current page')

		result = await self.controller.multi_act(updated_actions, self.browser_context)

		await asyncio.sleep(delay)
		return result

	async def _update_action_indices(
		self,
		historical_element: Optional[DOMHistoryElement],
		action: ActionModel,  # Type this properly based on your action model
		current_state: BrowserState,
	) -> Optional[ActionModel]:
		"""
		Update action indices based on current page state.
		Returns updated action or None if element cannot be found.
		"""
		if not historical_element or not current_state.element_tree:
			return action

		current_element = HistoryTreeProcessor.find_history_element_in_tree(
			historical_element, current_state.element_tree
		)

		if not current_element or current_element.highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != current_element.highlight_index:
			action.set_index(current_element.highlight_index)
			logger.info(
				f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}'
			)

		return action

	async def load_and_rerun(
		self, history_file: Optional[str | Path] = None, **kwargs
	) -> list[ActionResult]:
		"""
		Load history from file and rerun it.

		Args:
		        history_file: Path to the history file
		        **kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: Optional[str | Path] = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path)

	def create_history_gif(
		self,
		output_path: str = 'agent_history.gif',
		duration: int = 3000,
		show_goals: bool = True,
		show_task: bool = True,
		show_logo: bool = True,
		font_size: int = 40,
		title_font_size: int = 56,
		goal_font_size: int = 44,
		margin: int = 40,
		line_spacing: float = 1.5,
	) -> None:
		"""Create a GIF from the agent's history with overlaid task and goal text."""
		if not self.history.history:
			logger.warning('No history to create GIF from')
			return

		images = []

		# Try to load nicer fonts
		try:
			# Try different font options in order of preference
			font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
			font_loaded = False

			for font_name in font_options:
				try:
					regular_font = ImageFont.truetype(font_name, font_size)
					title_font = ImageFont.truetype(font_name, title_font_size)
					goal_font = ImageFont.truetype(font_name, goal_font_size)
					font_loaded = True
					break
				except OSError:
					continue

			if not font_loaded:
				raise OSError('No preferred fonts found')

		except OSError:
			regular_font = ImageFont.load_default()
			title_font = ImageFont.load_default()

			goal_font = regular_font

		# Load logo if requested
		logo = None
		if show_logo:
			try:
				logo = Image.open('./static/browser-use.png')
				# Resize logo to be small (e.g., 40px height)
				logo_height = 50
				aspect_ratio = logo.width / logo.height
				logo_width = int(logo_height * aspect_ratio)
				logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
			except Exception as e:
				logger.warning(f'Could not load logo: {e}')

		# Create task frame if requested
		if show_task and self.task:
			task_frame = self._create_task_frame(
				self.task,
				self.history.history[0].state.screenshot,
				title_font,
				regular_font,
				logo,
				line_spacing,
			)
			images.append(task_frame)

		# Process each history item
		for i, item in enumerate(self.history.history, 1):
			if not item.state.screenshot:
				continue

			# Convert base64 screenshot to PIL Image
			img_data = base64.b64decode(item.state.screenshot)
			image = Image.open(io.BytesIO(img_data))

			if show_goals and item.model_output:
				image = self._add_overlay_to_image(
					image=image,
					step_number=i,
					goal_text=item.model_output.current_state.next_goal,
					regular_font=regular_font,
					title_font=title_font,
					margin=margin,
					logo=logo,
				)

			images.append(image)

		if images:
			# Save the GIF
			images[0].save(
				output_path,
				save_all=True,
				append_images=images[1:],
				duration=duration,
				loop=0,
				optimize=False,
			)
			logger.info(f'Created history GIF at {output_path}')
		else:
			logger.warning('No images found in history to create GIF')

	def _create_task_frame(
		self,
		task: str,
		first_screenshot: str,
		title_font: ImageFont.FreeTypeFont,
		regular_font: ImageFont.FreeTypeFont,
		logo: Optional[Image.Image] = None,
		line_spacing: float = 1.5,
	) -> Image.Image:
		"""Create initial frame showing the task."""
		img_data = base64.b64decode(first_screenshot)
		template = Image.open(io.BytesIO(img_data))
		image = Image.new('RGB', template.size, (0, 0, 0))
		draw = ImageDraw.Draw(image)

		# Calculate vertical center of image
		center_y = image.height // 2

		# Draw "Task:" title
		title = 'Task:'
		title_bbox = draw.textbbox((0, 0), title, font=title_font)
		title_width = title_bbox[2] - title_bbox[0]
		title_x = (image.width - title_width) // 2
		title_y = center_y - 150  # Increased spacing from center

		draw.text(
			(title_x, title_y),
			title,
			font=title_font,
			fill=(255, 255, 255),
		)

		# Draw task text with increased spacing
		margin = 80  # Increased margin
		max_width = image.width - (2 * margin)
		wrapped_text = self._wrap_text(task, regular_font, max_width)

		# Calculate line height with spacing
		line_height = regular_font.size * line_spacing

		# Split text into lines and draw with custom spacing
		lines = wrapped_text.split('\n')
		total_height = line_height * len(lines)

		# Start position for first line
		text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

		for line in lines:
			# Get line width for centering
			line_bbox = draw.textbbox((0, 0), line, font=regular_font)
			text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2

			draw.text(
				(text_x, text_y),
				line,
				font=regular_font,
				fill=(255, 255, 255),
			)
			text_y += line_height

		# Add logo if provided (top right corner)
		if logo:
			logo_margin = 20
			logo_x = image.width - logo.width - logo_margin

			image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

		return image

	def _add_overlay_to_image(
		self,
		image: Image.Image,
		step_number: int,
		goal_text: str,
		regular_font: ImageFont.FreeTypeFont,
		title_font: ImageFont.FreeTypeFont,
		margin: int,
		logo: Optional[Image.Image] = None,
	) -> Image.Image:
		"""Add step number and goal overlay to an image."""
		image = image.convert('RGBA')
		txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
		draw = ImageDraw.Draw(txt_layer)

		# Add step number (bottom left)
		step_text = str(step_number)
		step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
		step_width = step_bbox[2] - step_bbox[0]
		step_height = step_bbox[3] - step_bbox[1]

		# Position step number in bottom left
		x_step = margin + 10  # Slight additional offset from edge
		y_step = image.height - margin - step_height - 10  # Slight offset from bottom

		# Draw background for step number with larger padding
		padding = 20  # Increased padding
		step_bg_bbox = (
			x_step - padding,
			y_step - padding,
			x_step + step_width + padding,
			y_step + step_height + padding,
		)
		draw.rectangle(step_bg_bbox, fill=(0, 0, 0, 255))

		# Draw step number
		draw.text(
			(x_step, y_step),
			step_text,
			font=title_font,
			fill=(255, 255, 255, 255),
		)

		# Draw goal text (centered, bottom)
		max_width = image.width - (4 * margin)
		wrapped_goal = self._wrap_text(goal_text, title_font, max_width)
		goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
		goal_width = goal_bbox[2] - goal_bbox[0]
		goal_height = goal_bbox[3] - goal_bbox[1]

		# Center goal text horizontally, place above step number
		x_goal = (image.width - goal_width) // 2
		y_goal = y_step - goal_height - padding * 4  # More space between step and goal

		# Draw background for goal with larger padding
		padding_goal = 25  # Increased padding for goal
		goal_bg_bbox = (
			x_goal - padding_goal,
			y_goal - padding_goal,
			x_goal + goal_width + padding_goal,
			y_goal + goal_height + padding_goal,
		)
		draw.rectangle(goal_bg_bbox, fill=(0, 0, 0, 255))

		# Draw goal text
		draw.multiline_text(
			(x_goal, y_goal),
			wrapped_goal,
			font=title_font,
			fill=(255, 255, 255, 255),
			align='center',
		)

		# Add logo if provided (top right corner)
		if logo:
			logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
			logo_margin = 20
			logo_x = image.width - logo.width - logo_margin
			logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
			txt_layer = Image.alpha_composite(logo_layer, txt_layer)

		# Composite and convert
		result = Image.alpha_composite(image, txt_layer)
		return result.convert('RGB')

	def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
		"""
		Wrap text to fit within a given width.

		Args:
			text: Text to wrap
			font: Font to use for text
			max_width: Maximum width in pixels

		Returns:
			Wrapped text with newlines
		"""
		words = text.split()
		lines = []
		current_line = []

		for word in words:
			current_line.append(word)
			line = ' '.join(current_line)
			bbox = font.getbbox(line)
			if bbox[2] > max_width:
				if len(current_line) == 1:
					lines.append(current_line.pop())
				else:
					current_line.pop()
					lines.append(' '.join(current_line))
					current_line = [word]

		if current_line:
			lines.append(' '.join(current_line))

		return '\n'.join(lines)
