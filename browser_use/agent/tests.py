import pytest

from browser_use.agent.views import (
	ActionResult,
	AgentBrain,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
)
from browser_use.browser.views import BrowserState, TabInfo
from browser_use.controller.registry.service import Registry
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.views import ClickElementAction, DoneAction, ExtractPageContentAction
from browser_use.dom.views import DomContentItem, ProcessedDomContent


@pytest.fixture
def sample_browser_state():
	return BrowserState(
		url='https://example.com',
		title='Example Page',
		current_tab_handle='tab1',
		tabs=[TabInfo(handle='tab1', url='https://example.com', title='Example Page')],
		screenshot='screenshot1.png',
		items=[
			DomContentItem(index=1, text='Click me', is_text_only=False, depth=0),
			DomContentItem(index=2, text='Some text', is_text_only=True, depth=1),
		],
		selector_map={1: '//button[1]', 2: '//div[1]'},
	)


@pytest.fixture
def action_registry():
	registry = Registry()

	# Register the actions we need for testing
	@registry.action(
		description='Click an element', requires_browser=True, param_model=ClickElementAction
	)
	def click_element(params: ClickElementAction, browser=None):
		pass

	@registry.action(
		description='Extract page content',
		requires_browser=True,
		param_model=ExtractPageContentAction,
	)
	def extract_page_content(params: ExtractPageContentAction, browser=None):
		pass

	@registry.action(description='Mark task as done', param_model=DoneAction)
	def done(params: DoneAction):
		pass

	# Create the dynamic ActionModel with all registered actions
	return registry.create_action_model()


@pytest.fixture
def sample_history(action_registry):
	# Create actions with nested params structure
	click_action = action_registry(click_element={'index': 1, 'num_clicks': 1})

	extract_action = action_registry(extract_page_content={'value': 'text'})

	done_action = action_registry(done={'text': 'Task completed'})

	histories = [
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					valuation_previous_goal='None', memory='Started task', next_goal='Click button'
				),
				action=click_action,
			),
			result=ActionResult(is_done=False),
			state=BrowserState(
				url='https://example.com',
				title='Page 1',
				current_tab_handle='tab1',
				tabs=[TabInfo(handle='tab1', url='https://example.com', title='Page 1')],
				screenshot='screenshot1.png',
				items=[DomContentItem(index=1, text='Button', is_text_only=False, depth=0)],
				selector_map={1: '//button[1]'},
			),
		),
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					valuation_previous_goal='Clicked button',
					memory='Button clicked',
					next_goal='Extract content',
				),
				action=extract_action,
			),
			result=ActionResult(
				is_done=False,
				extracted_content='Extracted text',
				error='Failed to extract completely',
			),
			state=BrowserState(
				url='https://example.com/page2',
				title='Page 2',
				current_tab_handle='tab1',
				tabs=[TabInfo(handle='tab1', url='https://example.com/page2', title='Page 2')],
				screenshot='screenshot2.png',
				items=[DomContentItem(index=2, text='Content', is_text_only=True, depth=0)],
				selector_map={2: '//div[1]'},
			),
		),
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					valuation_previous_goal='Extracted content',
					memory='Content extracted',
					next_goal='Finish task',
				),
				action=done_action,
			),
			result=ActionResult(is_done=True, extracted_content='Task completed', error=None),
			state=BrowserState(
				url='https://example.com/page2',
				title='Page 2',
				current_tab_handle='tab1',
				tabs=[TabInfo(handle='tab1', url='https://example.com/page2', title='Page 2')],
				screenshot='screenshot3.png',
				items=[DomContentItem(index=3, text='Final', is_text_only=True, depth=0)],
				selector_map={3: '//div[2]'},
			),
		),
	]
	return AgentHistoryList(history=histories)


def test_last_model_output(sample_history: AgentHistoryList):
	last_output = sample_history.last_model_output()
	print(last_output)
	assert last_output == {'done': {'text': 'Task completed'}}


def test_get_errors(sample_history: AgentHistoryList):
	errors = sample_history.get_errors()
	assert len(errors) == 1
	assert errors[0] == 'Failed to extract completely'


def test_final_result(sample_history: AgentHistoryList):
	assert sample_history.final_result() == 'Task completed'


def test_is_done(sample_history: AgentHistoryList):
	assert sample_history.is_done() == True


def test_unique_urls(sample_history: AgentHistoryList):
	urls = sample_history.unique_urls()
	assert len(urls) == 2
	assert 'https://example.com' in urls
	assert 'https://example.com/page2' in urls


def test_all_screenshots(sample_history: AgentHistoryList):
	screenshots = sample_history.all_screenshots()
	assert len(screenshots) == 3
	assert screenshots == ['screenshot1.png', 'screenshot2.png', 'screenshot3.png']


def test_all_model_outputs(sample_history: AgentHistoryList):
	outputs = sample_history.all_model_outputs()
	assert len(outputs) == 3
	assert outputs[0] == {'click_element': {'index': 1, 'xpath': '//button[1]', 'num_clicks': 1}}
	assert outputs[1] == {'extract_page_content': {'value': 'text'}}
	assert outputs[2] == {'done': {'text': 'Task completed'}}


def test_all_model_outputs_filtered(sample_history: AgentHistoryList):
	filtered = sample_history.all_model_outputs_filtered(include=['click_element'])
	assert len(filtered) == 1
	assert filtered[0]['click_element']['index'] == 1


def test_empty_history():
	empty_history = AgentHistoryList(history=[])
	assert empty_history.last_model_output() is None
	assert empty_history.final_result() is None
	assert empty_history.is_done() == False
	assert len(empty_history.unique_urls()) == 0


# Add a test to verify action creation
def test_action_creation(action_registry):
	click_action = action_registry(click_element={'index': 1, 'num_clicks': 1})

	assert click_action.model_dump(exclude_none=True) == {
		'click_element': {'index': 1, 'num_clicks': 1}
	}


# run this with:
# pytest browser_use/agent/tests.py