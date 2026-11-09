import pytest
from browser_use.utils import is_placeholder_url, sanitize_url_candidate


def test_is_placeholder_url_xxx_patterns():
	assert is_placeholder_url('https://XXX.XX')
	assert is_placeholder_url('http://xxx.xxx.xxx')
	assert is_placeholder_url('https://www.xxxx.xxx')
	assert not is_placeholder_url('https://google.com')
	assert not is_placeholder_url('https://github.com/browser-use/browser-use')


def test_is_placeholder_url_reserved_tlds():
	# RFC 2606 / RFC 6761 reserved special-use TLDs
	assert is_placeholder_url('http://mytestsite.test')
	assert is_placeholder_url('https://api.myapp.example')
	assert is_placeholder_url('http://dummy.invalid')
	assert is_placeholder_url('http://service.localhost')
	assert is_placeholder_url('https://host.local')


def test_is_placeholder_url_reserved_domains():
	# Standard example domains
	assert is_placeholder_url('https://example.com')
	assert is_placeholder_url('http://www.example.org/path?q=1')
	assert is_placeholder_url('https://sub.example.net')
	assert is_placeholder_url('http://yourdomain.com')
	assert is_placeholder_url('https://placeholder.com/image.png')


def test_is_placeholder_url_invalid_or_empty():
	assert not is_placeholder_url('')
	assert not is_placeholder_url('   ')
	assert not is_placeholder_url('plain_text_without_host')


def test_sanitize_url_candidate_brackets_and_quotes():
	# Strips surrounding angle brackets
	assert sanitize_url_candidate('<https://example.com/test>') == 'https://example.com/test'

	# Strips surrounding backticks
	assert sanitize_url_candidate('`https://example.com/api`') == 'https://example.com/api'

	# Strips prose punctuation at the end
	assert sanitize_url_candidate('https://example.com/search.') == 'https://example.com/search'
	assert sanitize_url_candidate('https://example.com/search,') == 'https://example.com/search'
	assert sanitize_url_candidate('https://example.com/search;') == 'https://example.com/search'
	assert sanitize_url_candidate('https://example.com/search!') == 'https://example.com/search'
	assert sanitize_url_candidate('https://example.com/search?') == 'https://example.com/search'

	# Preserves balanced brackets in path
	assert (
		sanitize_url_candidate('https://en.wikipedia.org/wiki/Python_(programming_language)')
		== 'https://en.wikipedia.org/wiki/Python_(programming_language)'
	)

	# Strips trailing unbalanced bracket from enclosing sentence: "(see https://example.com)"
	assert sanitize_url_candidate('https://example.com)') == 'https://example.com'

	# Handles escaped newlines in task prose
	assert (
		sanitize_url_candidate('https://example.com/item\\n2. Next instructions')
		== 'https://example.com/item'
	)
