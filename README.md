# docusaurus-llm-translator

Translate Docusaurus MDX files using any LLM, without breaking formatting or outbound links.

requirements:

    pip install requests

Local use:

      python bin/translate.py  --source-docs-dir ./docs \
        --i18n-root ./i18n --cache-file tr_cache.json \
        --languages fr --docs-plugin-path "docusaurus-plugin-content-docs/current" \
        file.mdx file2.mdx

Or use with a Github Action, similar to:

```yaml
name: Auto Translate i18n
on:
  push:
    paths: ['docs/**/*.md', 'docs/**/*.mdx']
    branches: [main]

jobs:
  translate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install requests

    - name: Download cache
      uses: actions/download-artifact@v4
      with:
        name: translation-cache
        path: .
      continue-on-error: true

    - name: Get changed files
      id: changed
      run: |
        echo "CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD | grep -E '\.(md|mdx)$' | tr '\n' ' ')" >> $GITHUB_ENV

    - name: Run translation
      env:
        API_KEY: ${{ secrets.API_KEY }}
        API_ENDPOINT_URL: 'https://whatever/chat/completions'
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: python translate.py

    - name: Upload cache
      uses: actions/upload-artifact@v4
      with:
        name: translation-cache
        path: translation_cache.json
        retention-days: 30
```
