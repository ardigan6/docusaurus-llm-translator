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
name: Translate MDX Files

on:
  pull_request:
    branches: # Specify branches to target, e.g., [ main, master ]
      - main
    paths:
      - 'docs/**/*.mdx' # Only run if MDX files in docs/ are changed
  workflow_dispatch: # Allows manual triggering
    inputs:
      languages:
        description: 'Space-separated language codes (e.g., "fr es ja")'
        required: false
        default: 'fr es' # Default languages for manual run

permissions:
  contents: write # Required to push changes back to the PR branch
  pull-requests: read # Required to get PR details

jobs:
  translate_docs:
    runs-on: ubuntu-latest
    env:
      # Secrets for the translation script
      API_KEY: ${{ secrets.LLM_API_KEY }}
      API_ENDPOINT_URL: ${{ secrets.LLM_API_ENDPOINT_URL }} # Optional, if overriding script default
      TRANSLATE_MODEL_NAME: ${{ secrets.LLM_TRANSLATE_MODEL_NAME }} # Optional
      REVIEW_MODEL_NAME: ${{ secrets.LLM_REVIEW_MODEL_NAME }} # Optional

      # Configuration for the script arguments
      SOURCE_DOCS_DIR: "docs"
      I18N_ROOT_DIR: "i18n"
      DOCS_PLUGIN_PATH: "docusaurus-plugin-content-docs/current"
      CACHE_FILE_PATH: "tr_cache.json" # As per your example, ensure script arg matches
      # Set languages: Use manual input if provided, otherwise default.
      # For PR triggers, it will use the default 'fr es'. You can change this default.
      LANGUAGES: ${{ github.event.inputs.languages || 'fr es' }}

    steps:
      - name: Checkout PR Branch
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.ref }} # Checkout PR head or current ref for dispatch
          fetch-depth: 0 # Fetches all history for git diff

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: pip install requests # Add other dependencies if any

      - name: Cache Translation Data
        uses: actions/cache@v4
        id: trans-cache
        with:
          path: ${{ env.CACHE_FILE_PATH }}
          key: ${{ runner.os }}-translation-cache-${{ github.head_ref }}-${{ hashFiles(env.CACHE_FILE_PATH) }}
          restore-keys: |
            ${{ runner.os }}-translation-cache-${{ github.head_ref }}-
            ${{ runner.os }}-translation-cache-

      - name: Get Changed MDX Files
        id: changed_mdx_files
        run: |
          # For PRs, compare head with base. For manual dispatch, use all mdx files (or adjust logic)
          if [[ "${{ github.event_name }}" == "pull_request" ]]; then
            BASE_SHA="${{ github.event.pull_request.base.sha }}"
            HEAD_SHA="${{ github.event.pull_request.head.sha }}"
            # Get files that were Added (A), Modified (M), Copied (C), or Renamed (R)
            CHANGED_FILES_LIST=$(git diff --name-only --diff-filter=AMCR "$BASE_SHA" "$HEAD_SHA" -- "${{ env.SOURCE_DOCS_DIR }}/**/*.mdx" | tr '\n' ' ')
          else # For workflow_dispatch, consider all files or a specific subset
            echo "Workflow dispatch: processing all MDX files in ${{ env.SOURCE_DOCS_DIR }}"
            CHANGED_FILES_LIST=$(find "${{ env.SOURCE_DOCS_DIR }}" -type f -name "*.mdx" | tr '\n' ' ')
          fi

          if [[ -z "$CHANGED_FILES_LIST" ]]; then
            echo "No MDX files to process."
            echo "PROCESS_FILES=false" >> $GITHUB_ENV
          else
            echo "MDX files to process: $CHANGED_FILES_LIST"
            echo "PROCESS_FILES=true" >> $GITHUB_ENV
            echo "CHANGED_FILES_ARGS=$CHANGED_FILES_LIST" >> $GITHUB_ENV
          fi
        shell: bash

      - name: Run Translation Script
        if: env.PROCESS_FILES == 'true'
        run: |
          python3 bin/translate.py \
            --source-docs-dir "${{ env.SOURCE_DOCS_DIR }}" \
            --i18n-root "${{ env.I18N_ROOT_DIR }}" \
            --cache-file "${{ env.CACHE_FILE_PATH }}" \
            --languages ${{ env.LANGUAGES }} \
            --docs-plugin-path "${{ env.DOCS_PLUGIN_PATH }}" \
            ${{ env.CHANGED_FILES_ARGS }}
        # Add --verbose if needed:
        #   --verbose

      - name: Commit and Push Translations
        if: env.PROCESS_FILES == 'true'
        run: |
          git config --global user.name 'GitHub Actions Bot'
          git config --global user.email 'github-actions-bot@users.noreply.github.com'

          # Add all changes in the i18n directory and the cache file
          git add "${{ env.I18N_ROOT_DIR }}" "${{ env.CACHE_FILE_PATH }}"

          # Check if there are changes to commit
          if git diff --staged --quiet; then
            echo "No changes to commit."
          else
            git commit -m "Apply automatic translations

          Affected files by this PR processed:
          ${{ env.CHANGED_FILES_ARGS }}"
            # Use GITHUB_TOKEN for pushing to the PR branch
            git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git HEAD:${{ github.head_ref }}
          fi
        shell: bash
```
