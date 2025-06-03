# docusaurus-llm-translator

Translate Docusaurus MDX using any LLM.

Use with a Github Action similar to:

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
        LLM_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: python translate.py

    - name: Upload cache
      uses: actions/upload-artifact@v4
      with:
        name: translation-cache
        path: translation_cache.json
        retention-days: 30
```
