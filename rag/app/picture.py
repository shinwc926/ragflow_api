#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
import io
import re
import tempfile
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from api.db.services.llm_service import LLMBundle
from common.constants import LLMType
from common.string_utils import clean_markdown_block
from deepdoc.parser.pdf_parser import RAGFlowPdfParser
from deepdoc.vision import OCR
from rag.nlp import (
    attach_media_context,
    naive_merge,
    naive_merge_with_images,
    rag_tokenizer,
    tokenize,
    tokenize_chunks,
    tokenize_chunks_with_images,
)

ocr = OCR()

# Gemini supported MIME types
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".flv", ".mpeg", ".mpg", ".webm", ".wmv", ".3gp", ".3gpp", ".mkv"]


def _find_mineru_ocr_model(tenant_id, mineru_llm_name, lang, **kwargs):
    """
    Find and create MinerU OCR model via LLMBundle.
    
    Args:
        tenant_id: Tenant ID for LLMBundle
        mineru_llm_name: MinerU LLM name (optional)
        lang: Language code
        **kwargs: Additional arguments including parser_config
        
    Returns:
        Tuple of (pdf_parser, actual_lang) or (None, None) if failed
    """
    if not tenant_id:
        logging.warning("[Picture OCR] tenant_id is required for MinerU OCR")
        return None, None
    
    if not mineru_llm_name:
        try:
            from api.db.services.tenant_llm_service import TenantLLMService
            env_name = TenantLLMService.ensure_mineru_from_env(tenant_id)
            candidates = TenantLLMService.query(tenant_id=tenant_id, llm_factory="MinerU", model_type=LLMType.OCR)
            if candidates:
                mineru_llm_name = candidates[0].llm_name
            elif env_name:
                mineru_llm_name = env_name
        except Exception as e:
            logging.warning(f"[Picture OCR] Failed to get MinerU LLM name: {e}")
            return None, None
    
    if not mineru_llm_name:
        logging.warning("[Picture OCR] MinerU LLM not found")
        return None, None
    
    try:
        # Get mineru_lang from kwargs or use default
        parser_cfg = kwargs.get('parser_config', {})
        actual_lang = parser_cfg.get('mineru_lang', lang)
        # If not in parser_config, try to get from KB config
        if not actual_lang:
            kb_id = kwargs.get('kb_id')
            if kb_id:
                try:
                    from api.db.services.knowledgebase_service import KnowledgebaseService
                    e, kb = KnowledgebaseService.get_by_id(kb_id)
                    if e and kb and hasattr(kb, 'parser_config') and kb.parser_config:
                        actual_lang = kb.parser_config.get('mineru_lang')
                        logging.info(f"[naive.by_mineru] Got mineru_lang from KB parser_config: {actual_lang}")
                except Exception as e:
                    logging.warning(f"[naive.by_mineru] Failed to fetch KB parser_config: {e}")
        
        # Final fallback to lang parameter
        if not actual_lang:
            actual_lang = lang    
            
        # Create LLMBundle OCR model
        ocr_model = LLMBundle(tenant_id=tenant_id, llm_type=LLMType.OCR, llm_name=mineru_llm_name, lang=actual_lang)
        pdf_parser = ocr_model.mdl
        
        return pdf_parser, actual_lang
        
    except Exception as e:
        logging.warning(f"[Picture OCR] Failed to create MinerU OCR model: {e}")
        import traceback
        logging.warning(traceback.format_exc())
        return None, None


def _ocr_with_mineru(filename, binary, tenant_id, lang, mineru_llm_name=None, callback=None, **kwargs):
    """
    Use MinerU to perform OCR on an image with language support via LLMBundle.
    
    Args:
        filename: Original filename for the image
        binary: Image binary data (bytes)
        tenant_id: Tenant ID for LLMBundle
        lang: Language code (e.g., 'korean', 'english', 'chinese')
        mineru_llm_name: MinerU LLM name (optional)
        callback: Progress callback function
        
    Returns:
        Extracted text from the image, or None if failed
    """
    # Find MinerU OCR model
    pdf_parser, actual_lang = _find_mineru_ocr_model(tenant_id, mineru_llm_name, lang, **kwargs)
    if not pdf_parser:
        return None
    
    try:
        if callback:
            callback(0.1, f"[MinerU OCR] Processing with language: {actual_lang}")
        
        # Use parse_pdf method with binary directly (no conversion needed)
        sections, tables, images = pdf_parser.parse_pdf(
            filepath=filename,
            binary=binary,
            callback=callback,
            parse_method='ocr',
            lang=actual_lang,
            **kwargs
        )
        
        if callback:
            callback(0.8, "[MinerU OCR] Extraction complete")
        
        # Extract text from sections
        if sections:
            texts = [section[0] for section in sections if section and section[0]]
            if texts:
                return '\n\n'.join(texts)
        
        logging.warning("[Picture OCR] No text extracted from MinerU")
        return None
            
    except Exception as e:
        logging.warning(f"[Picture OCR] MinerU OCR failed: {e}")
        import traceback
        logging.warning(traceback.format_exc())
        return None


def _ocr_with_mineru_structured(filename, binary, tenant_id, lang, mineru_llm_name=None, callback=None, **kwargs):
    """
    Use MinerU to perform OCR on an image and extract structured content (sections, tables, images) via LLMBundle.
    
    Args:
        filename: Original filename for the image
        binary: Image binary data (bytes)
        tenant_id: Tenant ID for LLMBundle
        lang: Language code (e.g., 'korean', 'english', 'chinese')
        mineru_llm_name: MinerU LLM name (optional)
        callback: Progress callback function
        
    Returns:
        Tuple of (sections, tables, images):
        - sections: List[Tuple[str, str]] - (text_content, line_tag)
        - tables: List (currently empty, can be extended)
        - images: List[Tuple[Tuple[PIL.Image, str], List[Tuple]]] - ((image, caption), bbox_list)
        or None if failed
    """
    # Find MinerU OCR model
    pdf_parser, actual_lang = _find_mineru_ocr_model(tenant_id, mineru_llm_name, lang, **kwargs)
    if not pdf_parser:
        return None
    
    try:
        if callback:
            callback(0.1, f"[MinerU OCR Structured] Processing with language: {actual_lang}")
        
        # Use parse_pdf method with binary directly (no conversion needed)
        sections, tables, images = pdf_parser.parse_pdf(
            filepath=filename,
            binary=binary,
            callback=callback,
            parse_method='ocr',
            lang=actual_lang,
            **kwargs
        )
        
        if callback:
            callback(0.9, f"[MinerU OCR Structured] Complete: {len(sections)} sections, {len(tables)} tables, {len(images)} images")
        
        logging.info(f"[Picture OCR Structured] Extracted: {len(sections)} sections, {len(tables)} tables, {len(images)} images")
        return (sections, tables, images)
            
    except Exception as e:
        logging.warning(f"[Picture OCR Structured] MinerU structured OCR failed: {e}")
        import traceback
        logging.warning(traceback.format_exc())
        return None


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)),
    }
    eng = lang.lower() == "english"
    use_structured_ocr = True

    parser_config = kwargs.get("parser_config", {}) or {}
    image_ctx = max(0, int(parser_config.get("image_context_size", 0) or 0))
    mineru_llm_name = kwargs.get("mineru_llm_name")

    if any(filename.lower().endswith(ext) for ext in VIDEO_EXTS):
        try:
            doc.update(
                {
                    "doc_type_kwd": "video",
                }
            )
            cv_mdl = LLMBundle(tenant_id, llm_type=LLMType.IMAGE2TEXT, lang=lang)
            ans = asyncio.run(
                cv_mdl.async_chat(system="", history=[], gen_conf={}, video_bytes=binary, filename=filename))
            callback(0.8, "CV LLM respond: %s ..." % ans[:32])
            ans += "\n" + ans
            tokenize(doc, ans, eng)
            return [doc]
        except Exception as e:
            callback(prog=-1, msg=str(e))
    else:
        # Don't convert to PIL Image yet - pass binary directly to MinerU
        img = Image.open(io.BytesIO(binary)).convert("RGB")
        doc.update(
            {
                "doc_type_kwd": "image",
                "image": img
            }
        )
        
        # Option 1: Structured OCR (extracts sections, tables, images)
        if use_structured_ocr:
            result = _ocr_with_mineru_structured(filename, binary, tenant_id, lang, mineru_llm_name, callback, **kwargs)
            if result:
                sections, tables, images = result
                
                if callback:
                    callback(0.7, f"Structured OCR complete: {len(sections)} sections, {len(tables)} tables, {len(images)} images")
                
                # Extract section images
                section_images = [img[0][0] if img and img[0] else None for img in images] if images else None
                
                # Keep sections with position tags as individual chunks (no merging)
                # Each section already has position tag information in format: "text@@page\tx\ty\tw\th##"
                chunks = []
                chunk_images = []
                
                for idx, sec in enumerate(sections):
                    text, position_tag = sec if isinstance(sec, tuple) else (sec, "")
                    
                    # Keep the full text with position tag intact
                    chunk_text = text + position_tag
                    chunks.append(chunk_text)
                    
                    # Associate image if available
                    if section_images and idx < len(section_images):
                        chunk_images.append(section_images[idx])
                    else:
                        chunk_images.append(None)
                
                if callback:
                    callback(0.85, f"Created {len(chunks)} chunks (block-based with position tags)")
                
                # Remove position tags before tokenizing
                chunks_clean = [RAGFlowPdfParser.remove_tag(ck) for ck in chunks]
                
                # Tokenize chunks with images if available
                if chunk_images and any(img is not None for img in chunk_images):
                    results = tokenize_chunks_with_images(chunks_clean, doc, eng, chunk_images)
                else:
                    results = tokenize_chunks(chunks_clean, doc, eng)
                
                return results
        
        # Option 2: Simple OCR (text only, original behavior)
        # Try MinerU OCR first (supports Korean) - pass binary directly
        txt = _ocr_with_mineru(filename, binary, tenant_id, lang, mineru_llm_name, callback, **kwargs)
        
        # Fallback to deepdoc OCR if MinerU fails (needs PIL Image)
        if not txt:
            callback(0.2, "Using fallback OCR...")
            bxs = ocr(np.array(img))
            txt = "\n".join([t[0] for _, t in bxs if t[0]])
        
        callback(0.4, "Finish OCR: (%s ...)" % txt[:12])
        if (eng and len(txt.split()) > 32) or len(txt) > 32:
            tokenize(doc, txt, eng)
            callback(0.8, "OCR results is too long to use CV LLM.")
            return attach_media_context([doc], 0, image_ctx)

        try:
            callback(0.4, "Use CV LLM to describe the picture.")
            cv_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, lang=lang)
            img_binary = io.BytesIO()
            img.save(img_binary, format="JPEG")
            img_binary.seek(0)
            ans = cv_mdl.describe(img_binary.read())
            callback(0.8, "CV LLM respond: %s ..." % ans[:32])
            txt += "\n" + ans
            tokenize(doc, txt, eng)
            return attach_media_context([doc], 0, image_ctx)
        except Exception as e:
            callback(prog=-1, msg=str(e))

    return []


def vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """
    A simple wrapper to process image to markdown texts via VLM.

    Returns:
        Simple markdown texts generated by VLM.
    """
    callback = callback or (lambda prog, msg: None)

    img = binary
    txt = ""

    try:
        with io.BytesIO() as img_binary:
            try:
                img.save(img_binary, format="JPEG")
            except Exception:
                img_binary.seek(0)
                img_binary.truncate()
                img.save(img_binary, format="PNG")

            img_binary.seek(0)
            ans = clean_markdown_block(vision_model.describe_with_prompt(img_binary.read(), prompt))
            txt += "\n" + ans
            return txt

    except Exception as e:
        callback(-1, str(e))

    return ""
