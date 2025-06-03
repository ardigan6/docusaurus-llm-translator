# example:
#
# python3 bin/translate.py  --source-docs-dir ./docs --i18n-root ./i18n --cache-file tr_cache.json --languages fr --docs-plugin-path "docusaurus-plugin-content-docs/current"  file.mdx

import re
import json
import hashlib
import os
import shutil
import requests
import argparse
import logging
import sys
from typing import List, Dict, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Environment variables and API configuration
API_KEY = os.environ.get("API_KEY") # Generic API Key name
# Endpoint example for OpenAI-compatible API
API_ENDPOINT_URL = os.environ.get("API_ENDPOINT_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
# Model names (can be the same if the model handles both tasks with different prompts)
TRANSLATE_MODEL_NAME = os.environ.get("TRANSLATE_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
REVIEW_MODEL_NAME = os.environ.get("REVIEW_MODEL_NAME", "gemini-2.5-flash-preview-05-20")

MAX_LLM_ATTEMPTS = 2 # Initial attempt + 1 retry

# Default paths, can be overridden by CLI args
DEFAULT_CACHE_FILE = "translation_cache.json"
DEFAULT_I18N_ROOT_DIR = "i18n"
DEFAULT_DOCS_PLUGIN_PATH = "docusaurus-plugin-content-docs/current"
DEFAULT_SOURCE_DOCS_DIR = "docs"

# noisy
DEBUG_VERBOSE = True

# Regex patterns
INLINE_CODE_REGEX = re.compile(r"`(.*?)`")
FENCED_CODE_BLOCK_REGEX = re.compile(r"^(```|~~~).*?\n(.*?\n)*?^\1", re.MULTILINE | re.DOTALL)
FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?\n)*?---(\s*\n|$)", re.MULTILINE | re.DOTALL)
HTML_COMMENT_REGEX = re.compile(r"<!--.*?-->", re.DOTALL)
HEADER_REGEX = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
LIST_ITEM_REGEX = re.compile(r"^(\s*[-*]|\s*\d+\.)\s+(.*)$", re.MULTILINE)
TABLE_ROW_REGEX = re.compile(r"^\|(.+)\|$", re.MULTILINE)
TABLE_SEPARATOR_REGEX = re.compile(r"^\|\s*:?-+:?\s*\|.*$", re.MULTILINE)
BLOCKQUOTE_REGEX = re.compile(r"^>\s?(.*)$", re.MULTILINE)
JSON_MARKDOWN_REGEX = re.compile(r"```json\s*([\s\S]*?)\s*```")

JSX_COMPONENT_REGEX = re.compile(r"<[A-Z][a-zA-Z0-9]*(?:\s+[^>]*)?\s*/?>", re.DOTALL)
IMPORT_STATEMENT_REGEX = re.compile(r"^import\s+.*?;", re.MULTILINE | re.DOTALL)
IMPORT_NAMED_EXPORTS_REGEX = re.compile(r"^import\s+[^,\{]*\{([^}]+)\}", re.MULTILINE)


class LLMService:
    def __init__(self, api_key: Optional[str], endpoint_url: str):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        if not self.api_key:
            logging.warning("LLM_SERVICE: API Key not provided. LLM calls will fail.")
        if not self.endpoint_url:
            # This was noted as a potential place to raise an error.
            # For now, allows instantiation but _make_request will fail.
            logging.error("LLM_SERVICE: API Endpoint URL not provided.")


    def _make_request(self, model: str, messages: List[Dict[str, str]], reasoning_effort: str = "low") -> Optional[Dict[str, Any]]:
        if not self.api_key or not self.endpoint_url:
            logging.error("LLM_SERVICE: API key or endpoint URL is missing. Cannot make request.")
            return None

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if reasoning_effort:
             payload["reasoning_effort"] = reasoning_effort

        logging.debug(f"LLM_SERVICE: Sending request to {self.endpoint_url} with model {model}.")
        try:
            response = requests.post(self.endpoint_url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"LLM_SERVICE: API call to {self.endpoint_url} (model {model}) failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"LLM_SERVICE: Response status: {e.response.status_code}, content: {e.response.text}")
            return None

    def get_chat_completion_content(self, messages: List[Dict[str, str]], model: str, parse_json_output: bool = True) -> Optional[Any]:
        api_response = self._make_request(model, messages)
        if not api_response:
            return None

        try:
            content = api_response["choices"][0]["message"]["content"]
            logging.debug(f"LLM_SERVICE: Raw content from model {model}: {content[:500]}...")
            if parse_json_output:
                match = JSON_MARKDOWN_REGEX.search(content)
                json_str = match.group(1) if match else content
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as jde:
                    logging.error(f"LLM_SERVICE: Failed to decode JSON from content (model {model}). Error: {jde}. Content sample: {json_str[:200]}")
                    return None
            return content
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"LLM_SERVICE: Failed to parse LLM response structure (model {model}): {e}. Response: {api_response}")
            return None

    def _build_translation_prompt(self, segments_to_translate: List[Dict[str,str]], target_lang: str) -> str:
        return f"""You are an expert translation engine. Your task is to translate a batch of English text segments into {target_lang}.
Each segment is provided with an 'id'. Preserve any placeholders like `$CODEVAR[n]` exactly as they appear in the original text.

These segments are from a MDX file. IMPORTANT RULES:
- DO NOT translate import statements, component names, or anything inside angle brackets (< >) or backticks (` `)
- DO NOT translate JSX component syntax like <ThemeConfigurator live={{true}} />
- DO NOT translate variable names, function names, or code examples
- DO NOT translate content that appears to be code, even if it looks like English words

Please provide your response as a single JSON object with one key "translations", which holds a list of objects. Each object in the list must have the following structure:
{{"id": "segment_hash", "translated_text": "translated segment text in {target_lang}"}}

Example of your response format:
{{
  "translations": [
    {{"id": "hash1", "translated_text": "Translated text for hash1 with $CODEVAR[0] in {target_lang}"}},
    {{"id": "hash2", "translated_text": "Another translated text for hash2 in {target_lang}"}}
  ]
}}

Ensure your output is only the JSON object, with no preceding or succeeding text.
The texts to translate are:
{json.dumps(segments_to_translate, indent=2, ensure_ascii=False)}"""

    def translate_batch(self, segments: List[Dict[str, str]], target_lang: str, model: str) -> Dict[str, str]:
        final_translations: Dict[str, str] = {}
        input_segments_map: Dict[str, Dict[str, str]] = {s["id"]: s for s in segments}
        ids_to_process: List[str] = [s["id"] for s in segments]

        for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
            if not ids_to_process: break

            current_batch_segments = [input_segments_map[id_val] for id_val in ids_to_process]
            logging.info(f"LLM_SERVICE: Translate attempt {attempt_num}/{MAX_LLM_ATTEMPTS} for {len(current_batch_segments)} segments, lang {target_lang}, model {model}.")

            prompt = self._build_translation_prompt(current_batch_segments, target_lang)
            messages = [{"role": "user", "content": prompt}]
            response_data = self.get_chat_completion_content(messages, model=model, parse_json_output=True)

            successfully_processed_ids_this_attempt = []
            if response_data and "translations" in response_data and isinstance(response_data["translations"], list):
                for item in response_data["translations"]:
                    segment_id = item.get("id")
                    translated_text = item.get("translated_text")
                    if segment_id and translated_text is not None and segment_id in input_segments_map:
                        final_translations[segment_id] = translated_text
                        successfully_processed_ids_this_attempt.append(segment_id)

            ids_to_process = [id_val for id_val in ids_to_process if id_val not in successfully_processed_ids_this_attempt]

            if not ids_to_process:
                logging.info(f"LLM_SERVICE: Translate successfully processed all segments for model {model} ({target_lang}) by attempt {attempt_num}.")
                break

            if attempt_num < MAX_LLM_ATTEMPTS:
                logging.warning(f"LLM_SERVICE: Translate attempt {attempt_num} for model {model} ({target_lang}) did not return translations for {len(ids_to_process)} IDs: {ids_to_process}. Retrying.")
            else:
                logging.error(f"LLM_SERVICE: Translate final attempt for model {model} ({target_lang}): Still missing translations for {len(ids_to_process)} IDs: {ids_to_process} after {MAX_LLM_ATTEMPTS} attempts.")
        return final_translations

    def _build_review_prompt(self, items_for_review: List[Dict[str, Any]], target_lang: str) -> str:
        return f"""You are an expert linguistic reviewer. Your task is to review a batch of translations from English to {target_lang}.
For each item, you are given a 'hash', the 'original_english' text, and the initial 'translated_by_llm1' text.
Both English and translated texts may contain placeholders like `$CODEVAR[n]`. These placeholders MUST be preserved in your 'final_translation'.

These segments are from a MDX file. IMPORTANT RULES:
- DO NOT translate import statements, component names, or anything inside angle brackets (< >) or backticks (` `)
- DO NOT translate JSX component syntax like <ThemeConfigurator live={{true}} />
- DO NOT translate variable names, function names, or code examples
- DO NOT translate content that appears to be code, even if it looks like English words

Please provide your response as a single JSON object with one key "reviews", which holds a list of objects. Each object in the list must have the following structure:
{{
  "hash": "item_hash",
  "approved": boolean,
  "final_translation": "The approved or corrected translation in {target_lang}. This MUST contain placeholders if present in original.",
  "notes": "A brief explanation if a correction was made. Optional."
}}

Example of your response format:
{{
  "reviews": [
    {{
      "hash": "hash1", "approved": false,
      "final_translation": "Corrected translation for hash1 with $CODEVAR[0] in {target_lang}",
      "notes": "Initial translation had a grammatical error."
    }},
    {{
      "hash": "hash2", "approved": true,
      "final_translation": "Initial translation for hash2 (because it was approved)",
      "notes": "Excellent translation."
    }}
  ]
}}

Ensure your output is only the JSON object, with no preceding or succeeding text.
The items to review are:
{json.dumps(items_for_review, indent=2, ensure_ascii=False)}"""

    def review_batch(self, items_for_review: List[Dict[str, Any]], target_lang: str, model: str) -> Dict[str, Dict[str, Any]]:
        final_reviews: Dict[str, Dict[str, Any]] = {}
        input_items_map: Dict[str, Dict[str, Any]] = {item["hash"]: item for item in items_for_review}
        hashes_to_process: List[str] = [item["hash"] for item in items_for_review]

        for attempt_num in range(1, MAX_LLM_ATTEMPTS + 1):
            if not hashes_to_process: break

            current_batch_items = [input_items_map[h] for h in hashes_to_process]
            logging.info(f"LLM_SERVICE: Review attempt {attempt_num}/{MAX_LLM_ATTEMPTS} for {len(current_batch_items)} items, lang {target_lang}, model {model}.")

            prompt = self._build_review_prompt(current_batch_items, target_lang)
            messages = [{"role": "user", "content": prompt}]
            response_data = self.get_chat_completion_content(messages, model=model, parse_json_output=True)

            successfully_processed_hashes_this_attempt = []
            if response_data and "reviews" in response_data and isinstance(response_data["reviews"], list):
                for review_item in response_data["reviews"]:
                    item_hash = review_item.get("hash")
                    final_translation = review_item.get("final_translation")
                    approved_status = review_item.get("approved")

                    if item_hash and final_translation is not None and approved_status is not None and item_hash in input_items_map:
                        final_reviews[item_hash] = {
                            "approved": approved_status,
                            "final_translation": final_translation,
                            "notes": review_item.get("notes", "")
                        }
                        successfully_processed_hashes_this_attempt.append(item_hash)

            hashes_to_process = [h for h in hashes_to_process if h not in successfully_processed_hashes_this_attempt]

            if not hashes_to_process:
                logging.info(f"LLM_SERVICE: Review successfully processed all items for model {model} ({target_lang}) by attempt {attempt_num}.")
                break

            if attempt_num < MAX_LLM_ATTEMPTS:
                logging.warning(f"LLM_SERVICE: Review attempt {attempt_num} for model {model} ({target_lang}) did not return reviews for {len(hashes_to_process)} hashes: {hashes_to_process}. Retrying.")
            else:
                logging.error(f"LLM_SERVICE: Review final attempt for model {model} ({target_lang}): Still missing reviews for {len(hashes_to_process)} hashes: {hashes_to_process} after {MAX_LLM_ATTEMPTS} attempts.")
        return final_reviews

# --- Utility Functions ---
def load_cache(cache_file_path: str) -> Dict[str, Dict[str, str]]:
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f: return json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Cache file {cache_file_path} is corrupted. Starting with empty cache.")
    return {}

def save_cache(data: Dict[str, Dict[str, str]], cache_file_path: str):
    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e: logging.error(f"Failed to save cache to {cache_file_path}: {e}")

def replace_code_with_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    code_map: Dict[str,str] = {}
    idx = 0
    def repl(matchobj):
        nonlocal idx
        code, placeholder = matchobj.group(0), f"$CODEVAR[{idx}]"
        code_map[placeholder] = code
        idx += 1
        return placeholder
    return INLINE_CODE_REGEX.sub(repl, text), code_map

def generate_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def restore_code_in_translated_text(translated_text_with_placeholders: str, code_map: Dict[str, str]) -> str:
    restored_text = translated_text_with_placeholders
    for placeholder, actual_code in code_map.items():
        # Ensure placeholder is treated as a literal string in replacement
        restored_text = restored_text.replace(placeholder, actual_code)
    return restored_text

def should_skip_segment(text: str) -> bool:
    """Check if a text segment should be skipped from translation."""
    text = text.strip()
    # Skip JSX components (like <ThemeConfigurator live={true} />)
    if JSX_COMPONENT_REGEX.search(text):
        return True

    # Skip import statements completely
    if IMPORT_STATEMENT_REGEX.search(text):
        return True

    # Skip code blocks and inline code that might be extracted
    if text.startswith('```') or (text.startswith('`') and text.endswith('`')):
        return True

    # Skip if text looks like named imports (contains curly braces with comma-separated identifiers)
    if IMPORT_NAMED_EXPORTS_REGEX.search(text): # This usually applies to lines like "import { Foo, Bar } from 'baz';"
        return True

    # Skip if text consists ENTIRELY of typical programming identifiers (e.g. "ComponentA, ComponentB" or "myVariable")
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*', text.strip()):
        return True

# --- Core Logic Functions ---
def extract_translatable_segments(mdx_content: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    # Extract and process frontmatter separately
    frontmatter_match = FRONTMATTER_REGEX.search(mdx_content)
    frontmatter_segments = []
    if frontmatter_match:
        frontmatter_content = frontmatter_match.group(0)
        # Look for title in frontmatter
        for line in frontmatter_content.split('\n'):
            line = line.strip()
            if line.startswith('title:'):
                title_text = line[6:].strip()  # Remove 'title:' prefix
                # Remove quotes if present
                if (title_text.startswith('"') and title_text.endswith('"')) or \
                   (title_text.startswith("'") and title_text.endswith("'")):
                    title_text = title_text[1:-1]
                if title_text and len(title_text) >= 5:
                    text_for_llm, code_map = replace_code_with_placeholders(title_text)
                    h = generate_hash(text_for_llm)
                    frontmatter_segments.append({
                        "original_raw": title_text,
                        "text_for_llm": text_for_llm,
                        "hash": h,
                        "code_map": code_map,
                        "is_frontmatter_title": True
                    })

    content_no_frontmatter = FRONTMATTER_REGEX.sub("", mdx_content)

    # Remove import statements first to prevent them from interfering with other processing
    content_no_imports = IMPORT_STATEMENT_REGEX.sub("", content_no_frontmatter)


    content_no_code_blocks = FENCED_CODE_BLOCK_REGEX.sub("", content_no_imports)
    content_no_comments = HTML_COMMENT_REGEX.sub("", content_no_code_blocks)
    content_no_jsx = JSX_COMPONENT_REGEX.sub("", content_no_comments)
    content_no_imports = IMPORT_STATEMENT_REGEX.sub("", content_no_jsx)

    lines = content_no_imports.splitlines()
    extracted_raw_texts: List[str] = []
    current_paragraph_lines: List[str] = []

    def process_current_paragraph():
        nonlocal current_paragraph_lines
        if current_paragraph_lines:
            para_text = "\n".join(current_paragraph_lines).strip()
            if para_text: extracted_raw_texts.append(para_text)
            current_paragraph_lines = []

    for raw_line in lines:
        stripped_line = raw_line.strip()
        if not stripped_line:
            process_current_paragraph(); continue

        # Skip lines that are part of import statements
        # if IMPORT_STATEMENT_REGEX.match(raw_line.strip()):
        #     process_current_paragraph(); continue

        header_match = HEADER_REGEX.match(raw_line)
        list_match = LIST_ITEM_REGEX.match(raw_line)
        blockquote_match = BLOCKQUOTE_REGEX.match(raw_line)

        if TABLE_SEPARATOR_REGEX.match(stripped_line): process_current_paragraph()
        elif header_match: process_current_paragraph(); extracted_raw_texts.append(header_match.group(2).strip())
        elif list_match: process_current_paragraph(); extracted_raw_texts.append(list_match.group(2).strip())
        elif blockquote_match: process_current_paragraph(); extracted_raw_texts.append(blockquote_match.group(1).strip())
        elif TABLE_ROW_REGEX.match(stripped_line):
            process_current_paragraph()
            cells = [cell.strip() for cell in stripped_line[1:-1].split('|')]
            extracted_raw_texts.extend(filter(None, cells))
        else: current_paragraph_lines.append(stripped_line)
    process_current_paragraph()

    seen_hashes = set()
    for text in extracted_raw_texts:
        text = text.strip()
        if not text or text.startswith("!") or text.isnumeric() or len(text) < 5: continue

        # Skip segments that shouldn't be translated
        if should_skip_segment(text):
            continue

        text_for_llm, code_map = replace_code_with_placeholders(text)
        check_text_empty = text_for_llm
        for ph_val in code_map.keys(): check_text_empty = check_text_empty.replace(ph_val, "")
        if not check_text_empty.strip(): continue
        h = generate_hash(text_for_llm)
        if h not in seen_hashes:
            segments.append({"original_raw": text, "text_for_llm": text_for_llm, "hash": h, "code_map": code_map})
            seen_hashes.add(h)

    # Combine frontmatter and content segments
    return frontmatter_segments + segments


def process_mdx_file(
    mdx_filepath: str, target_langs: List[str], cache: Dict[str, Dict[str, str]],
    llm_service: LLMService,
    source_docs_dir: str, i18n_root_dir: str, docs_plugin_path: str
):
    logging.info(f"Processing MDX file: {mdx_filepath}")
    try:
        with open(mdx_filepath, 'r', encoding='utf-8') as f: original_mdx_content = f.read()
    except Exception as e: logging.error(f"Error reading file {mdx_filepath}: {e}"); return

    extracted_segments = extract_translatable_segments(original_mdx_content)
    if not extracted_segments: logging.info(f"No translatable segments found in {mdx_filepath}."); return

    # Sort by length of original_raw (descending) to prioritize replacing longer strings first.
    # This can help avoid issues where a shorter segment is a substring of a longer one.
    extracted_segments.sort(key=lambda s: len(s['original_raw']), reverse=True)

    for lang in target_langs:
        logging.info(f"Starting translation of {mdx_filepath} to {lang}...")

        segments_to_translate_api: List[Dict[str,str]] = []
        # This dictionary will hold the final form of translations (text_for_llm) for the current file and language,
        # after caching, translation, and review.
        current_file_final_translations: Dict[str, str] = {}

        for seg_info in extracted_segments:
            h, text_for_llm = seg_info["hash"], seg_info["text_for_llm"]
            if h in cache and lang in cache[h]: # Cache hit
                current_file_final_translations[h] = cache[h][lang]
            else: # Not in cache, needs translation
                segments_to_translate_api.append({"id": h, "text": text_for_llm})

        if segments_to_translate_api:
            logging.info(f"Found {len(segments_to_translate_api)} segments needing remote translation to {lang} for {mdx_filepath}.")
            if DEBUG_VERBOSE:
                logging.info(f"segments_to_translate_api: {segments_to_translate_api}")

            llm1_translations = llm_service.translate_batch(segments_to_translate_api, lang, TRANSLATE_MODEL_NAME)

            if llm1_translations:
                items_for_review: List[Dict[str, Any]] = []
                hash_to_original_eng_text_for_llm = {s["id"]:s["text"] for s in segments_to_translate_api}

                for h, translated_text in llm1_translations.items():
                    original_english_for_review = hash_to_original_eng_text_for_llm.get(h)
                    if original_english_for_review:
                        items_for_review.append({
                            "hash": h,
                            "original_english": original_english_for_review,
                            "translated_by_llm1": translated_text
                        })
                    else:
                        # This case means LLM1 returned a hash not in the original request to it for this batch,
                        # or internal lookup failed. Use LLM1 directly as a fallback.
                        logging.warning(f"Could not find original English text for hash {h} (lang {lang}) for review. Using LLM1 translation directly for this segment.")
                        current_file_final_translations[h] = translated_text
                        if h not in cache: cache[h] = {}
                        cache[h][lang] = translated_text # Update cache with LLM1's version

                if items_for_review:
                    reviewed_translations_map = llm_service.review_batch(items_for_review, lang, REVIEW_MODEL_NAME)

                    # Iterate based on what LLM1 successfully translated, then check for review
                    for h, llm1_text_from_batch in llm1_translations.items():
                        final_text_for_segment: Optional[str] = None
                        if h in reviewed_translations_map:
                            review_data = reviewed_translations_map[h]
                            final_text_for_segment = review_data["final_translation"]
                            logging.debug(f"Using reviewed translation for {h} ({lang}). Approved: {review_data['approved']}")
                        elif h in current_file_final_translations:
                            # Segment already handled (e.g., due to missing original_english_for_review)
                            continue
                        else:
                            # Segment translated by LLM1 but not present in review results (e.g. review LLM failed for this item)
                            logging.warning(f"Review missing for segment {h} ({lang}) after review batch. Using LLM1 translation as fallback.")
                            final_text_for_segment = llm1_text_from_batch # Fallback to LLM1's translation

                        if final_text_for_segment is not None:
                            current_file_final_translations[h] = final_text_for_segment
                            if h not in cache: cache[h] = {}
                            cache[h][lang] = final_text_for_segment # Update global cache with the final version
                elif llm1_translations: # llm1_translations has items, but none were prepared for review
                    logging.info(f"No items were prepared for review for language {lang} (possibly all failed original text lookup). Using LLM1 translations directly.")
                    for h, translated_text in llm1_translations.items():
                        if h not in current_file_final_translations: # Avoid overwriting if already handled
                            current_file_final_translations[h] = translated_text
                            if h not in cache: cache[h] = {}
                            cache[h][lang] = translated_text
            else:
                logging.error(f"Translation step yielded no translations for {lang} for {mdx_filepath} after all attempts.")

        # Apply translations to file
        try:
            relative_path = os.path.relpath(mdx_filepath, start=source_docs_dir)
        except ValueError:
            logging.error(f"File {mdx_filepath} not relative to source dir {source_docs_dir}. Skipping file write.")
            continue

        target_filepath = os.path.join(i18n_root_dir, lang, docs_plugin_path, relative_path)
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)

        modified_content = original_mdx_content

        # Update import paths for i18n structure
        def update_import_paths(content):
            def replace_import_path(match):
                full_match = match.group(0)
                # Update relative paths for i18n directory structure
                # Handle both single and double quotes, and different relative path depths
                if "'../src/" in full_match:
                    updated = full_match.replace("'../src/", "'../../../../src/")
                elif '"../src/' in full_match:
                    updated = full_match.replace('"../src/', '"../../../../src/')
                elif "'../../src/" in full_match:
                    updated = full_match.replace("'../../src/", "'../../../../src/")
                elif '"../../src/' in full_match:
                    updated = full_match.replace('"../../src/', '"../../../../src/')
                else:
                    updated = full_match

                return updated

            return IMPORT_STATEMENT_REGEX.sub(replace_import_path, content)

        modified_content = update_import_paths(modified_content)

        for seg_info in extracted_segments: # Iterate in original (length-sorted) order
            original_raw, h, code_map = seg_info["original_raw"], seg_info["hash"], seg_info["code_map"]
            is_frontmatter_title = seg_info.get("is_frontmatter_title", False)
            translated_llm_form = current_file_final_translations.get(h)
            if translated_llm_form is not None:
                final_segment = restore_code_in_translated_text(translated_llm_form, code_map)

                # Replace the original raw string with the final translated segment
                if is_frontmatter_title:
                    # Special handling for frontmatter title
                    # Replace the title value in frontmatter, preserving quotes if they were present
                    title_pattern = r'(title:\s*)(["\']?)(' + re.escape(original_raw) + r')(\2)'
                    def title_replacer(match):
                        return match.group(1) + match.group(2) + final_segment + match.group(4)
                    modified_content = re.sub(title_pattern, title_replacer, modified_content)
                else:
                    # Regular content replacement
                    modified_content = modified_content.replace(original_raw, final_segment)


        try:
            with open(target_filepath, 'w', encoding='utf-8') as f: f.write(modified_content)
            logging.info(f"Successfully wrote translated file: {target_filepath}")
        except IOError as e: logging.error(f"Failed to write {target_filepath}: {e}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Docusaurus i18n translation script using LLMs.")
    parser.add_argument("changed_files", nargs='+', help="List of changed MDX files.")
    parser.add_argument("--languages", nargs='+', default=["es", "fr"], help="Target language codes.")
    parser.add_argument("--source-docs-dir", default=DEFAULT_SOURCE_DOCS_DIR)
    parser.add_argument("--i18n-root", default=DEFAULT_I18N_ROOT_DIR)
    parser.add_argument("--docs-plugin-path", default=DEFAULT_DOCS_PLUGIN_PATH)
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    if not API_KEY:
        logging.error("API_KEY environment variable not set. LLM calls will not be made if translations are not cached.")
        # The script will proceed and use the cache if available.

    llm_service = LLMService(api_key=API_KEY, endpoint_url=API_ENDPOINT_URL)
    translation_cache = load_cache(args.cache_file)
    source_docs_dir_abs = os.path.abspath(args.source_docs_dir)

    for mdx_file_rel_path in args.changed_files:
        mdx_file_abs_path = os.path.abspath(mdx_file_rel_path)
        # Ensure the file path is correctly understood relative to the source_docs_dir
        if not mdx_file_abs_path.startswith(source_docs_dir_abs):
            potential_path = os.path.join(source_docs_dir_abs, mdx_file_rel_path)
            # Check if this constructed path is the correct one
            if os.path.exists(potential_path) and potential_path.startswith(source_docs_dir_abs):
                mdx_file_abs_path = potential_path
            else:
                logging.warning(f"File {mdx_file_rel_path} (resolved to {mdx_file_abs_path}) is not under source directory {source_docs_dir_abs} or does not exist. Skipping.")
                continue
        elif not os.path.exists(mdx_file_abs_path):
            logging.warning(f"File {mdx_file_abs_path} does not exist. Skipping.")
            continue

        process_mdx_file(
            mdx_file_abs_path, args.languages, translation_cache, llm_service,
            source_docs_dir=source_docs_dir_abs, i18n_root_dir=args.i18n_root,
            docs_plugin_path=args.docs_plugin_path
        )

    save_cache(translation_cache, args.cache_file)
    logging.info("Translation process finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
