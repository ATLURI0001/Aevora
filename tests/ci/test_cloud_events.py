import pytest

from browser_use.agent.cloud_events import CreateAgentOutputFileEvent, CreateAgentStepEvent, MAX_FILE_CONTENT_SIZE


def test_file_content_preserves_plain_content_with_commas():
    ev = CreateAgentOutputFileEvent(
        user_id='u1',
        task_id='t1',
        file_name='notes.csv',
        file_content='hello,world,123',
    )
    assert ev.file_content == 'hello,world,123'


def test_file_content_validates_data_uri_and_rejects_oversized_payload():
    ev = CreateAgentOutputFileEvent(
        user_id='u1',
        task_id='t1',
        file_name='tiny.txt',
        file_content='data:text/plain;base64,aGVsbG8=',
    )
    assert ev.file_content == 'data:text/plain;base64,aGVsbG8='

    oversized = 'A' * int((MAX_FILE_CONTENT_SIZE * 4 // 3) + 16)
    with pytest.raises(ValueError, match='exceeds maximum size'):
        CreateAgentOutputFileEvent(
            user_id='u1',
            task_id='t1',
            file_name='huge.bin',
            file_content=f'data:application/octet-stream;base64,{oversized}',
        )


def test_screenshot_data_uri_with_multiple_commas_validates_cleanly():
    step = CreateAgentStepEvent(
        user_id='u1',
        agent_task_id='t1',
        step=1,
        evaluation_previous_goal='ok',
        memory='ok',
        next_goal='ok',
        actions=[],
        screenshot_url='data:image/png;name=a,b;base64,aGVsbG8=',
    )
    assert step.screenshot_url is not None
