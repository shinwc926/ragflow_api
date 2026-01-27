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
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pdfplumber
import requests
from PIL import Image
from strenum import StrEnum

from deepdoc.parser.pdf_parser import RAGFlowPdfParser

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


class MinerUContentType(StrEnum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"
    CODE = "code"
    LIST = "list"
    DISCARDED = "discarded"


# Mapping from language names to MinerU language codes
LANGUAGE_TO_MINERU_MAP = {
    'English': 'en',
    'Chinese': 'ch',
    'Traditional Chinese': 'chinese_cht',
    'Russian': 'east_slavic',
    'Ukrainian': 'east_slavic',
    'Indonesian': 'latin',
    'Spanish': 'latin',
    'Vietnamese': 'latin',
    'Japanese': 'japan',
    'Korean': 'korean',
    'Portuguese BR': 'latin',
    'German': 'latin',
    'French': 'latin',
    'Italian': 'latin',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'ka',
    'Thai': 'th',
    'Greek': 'el',
    'Hindi': 'devanagari',
}


class MinerUBackend(StrEnum):
    """MinerU processing backend options."""

    PIPELINE = "pipeline"  # Traditional multimodel pipeline (default)
    VLM_TRANSFORMERS = "vlm-transformers"  # Vision-language model using HuggingFace Transformers
    VLM_MLX_ENGINE = "vlm-mlx-engine"  # Faster, requires Apple Silicon and macOS 13.5+
    VLM_VLLM_ENGINE = "vlm-vllm-engine"  # Local vLLM engine, requires local GPU
    VLM_VLLM_ASYNC_ENGINE = "vlm-vllm-async-engine"  # Asynchronous vLLM engine, new in MinerU API
    VLM_LMDEPLOY_ENGINE = "vlm-lmdeploy-engine"  # LMDeploy engine
    VLM_HTTP_CLIENT = "vlm-http-client"  # HTTP client for remote vLLM server (CPU only)


class MinerULanguage(StrEnum):
    """MinerU supported languages for OCR (pipeline backend only)."""

    CH = "ch"  # Chinese
    CH_SERVER = "ch_server"  # Chinese (server)
    CH_LITE = "ch_lite"  # Chinese (lite)
    EN = "en"  # English
    KOREAN = "korean"  # Korean
    JAPAN = "japan"  # Japanese
    CHINESE_CHT = "chinese_cht"  # Chinese Traditional
    TA = "ta"  # Tamil
    TE = "te"  # Telugu
    KA = "ka"  # Kannada
    TH = "th"  # Thai
    EL = "el"  # Greek
    LATIN = "latin"  # Latin
    ARABIC = "arabic"  # Arabic
    EAST_SLAVIC = "east_slavic"  # East Slavic
    CYRILLIC = "cyrillic"  # Cyrillic
    DEVANAGARI = "devanagari"  # Devanagari


class MinerUParseMethod(StrEnum):
    """MinerU PDF parsing methods (pipeline backend only)."""

    AUTO = "auto"  # Automatically determine the method based on the file type
    TXT = "txt"  # Use text extraction method
    OCR = "ocr"  # Use OCR method for image-based PDFs


@dataclass
class MinerUParseOptions:
    """Options for MinerU PDF parsing."""

    backend: MinerUBackend = MinerUBackend.PIPELINE
    lang: Optional[MinerULanguage] = None  # language for OCR (pipeline backend only)
    method: MinerUParseMethod = MinerUParseMethod.AUTO
    server_url: Optional[str] = None
    delete_output: bool = True
    parse_method: str = "raw"
    formula_enable: bool = True
    table_enable: bool = True


class MinerUParser(RAGFlowPdfParser):
    def __init__(self, mineru_path: str = "mineru", mineru_api: str = "", mineru_server_url: str = ""):
        self.mineru_api = mineru_api.rstrip("/")
        self.mineru_server_url = mineru_server_url.rstrip("/")
        self.outlines = []
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        return (member.external_attr >> 16) & 0o170000 == 0o120000

    def _extract_zip_no_root(self, zip_path, extract_to, root_dir):
        self.logger.info(f"[MinerU] Extract zip: zip_path={zip_path}, extract_to={extract_to}, root_hint={root_dir}")
        base_dir = Path(extract_to).resolve()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            if not root_dir:
                if members and members[0].filename.endswith("/"):
                    root_dir = members[0].filename
                else:
                    root_dir = None
            if root_dir:
                root_dir = root_dir.replace("\\", "/")
                if not root_dir.endswith("/"):
                    root_dir += "/"

            for member in members:
                if member.flag_bits & 0x1:
                    raise RuntimeError(f"[MinerU] Encrypted zip entry not supported: {member.filename}")
                if self._is_zipinfo_symlink(member):
                    raise RuntimeError(f"[MinerU] Symlink zip entry not supported: {member.filename}")

                name = member.filename.replace("\\", "/")
                if root_dir and name == root_dir:
                    self.logger.info("[MinerU] Ignore root folder...")
                    continue
                if root_dir and name.startswith(root_dir):
                    name = name[len(root_dir) :]
                if not name:
                    continue
                if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (absolute): {member.filename}")

                parts = [p for p in name.split("/") if p not in ("", ".")]
                if any(p == ".." for p in parts):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (traversal): {member.filename}")

                rel_path = os.path.join(*parts) if parts else ""
                dest_path = (Path(extract_to) / rel_path).resolve(strict=False)
                if dest_path != base_dir and base_dir not in dest_path.parents:
                    raise RuntimeError(f"[MinerU] Unsafe zip path (escape): {member.filename}")

                if member.is_dir():
                    os.makedirs(dest_path, exist_ok=True)
                    continue

                os.makedirs(dest_path.parent, exist_ok=True)
                with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _is_http_endpoint_valid(url, timeout=5):
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 301, 302, 307, 308]
        except Exception:
            return False

    def check_installation(self, backend: str = "pipeline", server_url: Optional[str] = None) -> tuple[bool, str]:
        reason = ""

        valid_backends = ["pipeline", "vlm-http-client", "vlm-transformers", "vlm-vllm-engine", "vlm-mlx-engine", "vlm-vllm-async-engine", "vlm-lmdeploy-engine"]
        if backend not in valid_backends:
            reason = f"[MinerU] Invalid backend '{backend}'. Valid backends are: {valid_backends}"
            self.logger.warning(reason)
            return False, reason

        if not self.mineru_api:
            reason = "[MinerU] MINERU_APISERVER not configured."
            self.logger.warning(reason)
            return False, reason

        api_openapi = f"{self.mineru_api}/openapi.json"
        try:
            api_ok = self._is_http_endpoint_valid(api_openapi)
            self.logger.info(f"[MinerU] API openapi.json reachable={api_ok} url={api_openapi}")
            if not api_ok:
                reason = f"[MinerU] MinerU API not accessible: {api_openapi}"
                return False, reason
        except Exception as exc:
            reason = f"[MinerU] MinerU API check failed: {exc}"
            self.logger.warning(reason)
            return False, reason

        if backend == "vlm-http-client":
            resolved_server = server_url or self.mineru_server_url
            if not resolved_server:
                reason = "[MinerU] MINERU_SERVER_URL required for vlm-http-client backend."
                self.logger.warning(reason)
                return False, reason
            try:
                server_ok = self._is_http_endpoint_valid(resolved_server)
                self.logger.info(f"[MinerU] vlm-http-client server check reachable={server_ok} url={resolved_server}")
            except Exception as exc:
                self.logger.warning(f"[MinerU] vlm-http-client server probe failed: {resolved_server}: {exc}")

        return True, reason

    def _run_mineru(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> Path:
        return self._run_mineru_api(input_path, output_dir, options, callback)

    def _run_mineru_api(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> Path:
        pdf_file_path = str(input_path)

        if not os.path.exists(pdf_file_path):
            raise RuntimeError(f"[MinerU] PDF file not exists: {pdf_file_path}")

        pdf_file_name = Path(pdf_file_path).stem.strip()
        output_path = tempfile.mkdtemp(prefix=f"{pdf_file_name}_{options.method}_", dir=str(output_dir))
        output_zip_path = os.path.join(str(output_dir), f"{Path(output_path).name}.zip")

        data = {
            "output_dir": "./output",
            "lang_list": options.lang,
            "backend": options.backend,
            "parse_method": options.method,
            "formula_enable": options.formula_enable,
            "table_enable": options.table_enable,
            "server_url": None,
            "return_md": True,
            "return_middle_json": True,
            "return_model_output": True,
            "return_content_list": True,
            "return_images": True,
            "response_format_zip": True,
            "start_page_id": 0,
            "end_page_id": 99999,
        }

        if options.server_url:
            data["server_url"] = options.server_url
        elif self.mineru_server_url:
            data["server_url"] = self.mineru_server_url

        self.logger.info(f"[MinerU] request {data=}")
        self.logger.info(f"[MinerU] request {options=}")

        headers = {"Accept": "application/json"}
        try:
            self.logger.info(f"[MinerU] invoke api: {self.mineru_api}/file_parse backend={options.backend} server_url={data.get('server_url')}")
            if callback:
                callback(0.20, f"[MinerU] invoke api: {self.mineru_api}/file_parse")
            with open(pdf_file_path, "rb") as pdf_file:
                files = {"files": (pdf_file_name + ".pdf", pdf_file, "application/pdf")}
                with requests.post(
                    url=f"{self.mineru_api}/file_parse",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=1800,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if content_type.startswith("application/zip"):
                        self.logger.info(f"[MinerU] zip file returned, saving to {output_zip_path}...")

                        if callback:
                            callback(0.30, f"[MinerU] zip file returned, saving to {output_zip_path}...")

                        with open(output_zip_path, "wb") as f:
                            response.raw.decode_content = True
                            shutil.copyfileobj(response.raw, f)

                        self.logger.info(f"[MinerU] Unzip to {output_path}...")
                        self._extract_zip_no_root(output_zip_path, output_path, pdf_file_name + "/")

                        if callback:
                            callback(0.40, f"[MinerU] Unzip to {output_path}...")
                    else:
                        self.logger.warning(f"[MinerU] not zip returned from api: {content_type}")
        except Exception as e:
            raise RuntimeError(f"[MinerU] api failed with exception {e}")
        self.logger.info("[MinerU] Api completed successfully.")
        return Path(output_path)

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=600, callback=None):
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.pdf = pdf
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for _, p in
                                    enumerate(self.pdf.pages[page_from:page_to])]
        except Exception as e:
            self.page_images = None
            self.total_page = 0
            self.logger.exception(e)

    def _line_tag(self, bx):
        pn = [bx["page_idx"] + 1]
        
        # bbox 키 존재 여부 확인 및 유효성 검증
        if "bbox" not in bx:
            self.logger.warning(f"[MinerU-BBOX] Block missing 'bbox' key: {bx.get('type', 'unknown')} on page {pn[0]}")
            # 기본 bbox 생성 (페이지 전체 영역으로 가정)
            positions = [0, 0, 100, 100]
        else:
            positions = bx["bbox"]
            # bbox가 None이거나 리스트가 아니면 기본값 사용
            if not positions or not isinstance(positions, list) or len(positions) != 4:
                self.logger.warning(f"[MinerU-BBOX] Invalid bbox format: {positions} on page {pn[0]}")
                positions = [0, 0, 100, 100]
        
        x0, top, x1, bott = positions

        # PDF 페이지의 실제 크기 사용 (이미지 크기가 아닌)
        if hasattr(self, "pdf") and self.pdf and self.pdf.pages and bx["page_idx"] < len(self.pdf.pages):
            pdf_page = self.pdf.pages[bx["page_idx"]]
            page_width = pdf_page.width
            page_height = pdf_page.height
            
            # 디버그: 첫 10개 블록과 100의 배수 페이지만 로그 출력
            if bx["page_idx"] < 10 or bx["page_idx"] % 100 == 0:
                image_idx = bx["page_idx"] - self.page_from
                if 0 <= image_idx < len(self.page_images):
                    img_size = self.page_images[image_idx].size
                    self.logger.debug(f"[MinerU-BBOX] page_idx={bx['page_idx']}, "
                                   f"PDF_size=({page_width:.1f}x{page_height:.1f}), "
                                   f"Image_size={img_size}, "
                                   f"bbox_norm=({x0:.1f},{top:.1f},{x1:.1f},{bott:.1f})")
            
            x0 = (x0 / 1000.0) * page_width
            x1 = (x1 / 1000.0) * page_width
            top = (top / 1000.0) * page_height
            bott = (bott / 1000.0) * page_height

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), x0, x1, top, bott)

    def crop(self, text, ZM=1, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            self.logger.warning("[MinerU] crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                self.logger.warning("[MinerU] Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                self.logger.warning(f"[MinerU] All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            self.logger.warning("[MinerU] No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            self.logger.warning(
                f"[MinerU] Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1]
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width

            if bottom <= top:
                bottom = top + 2

            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    self.logger.warning(
                        f"[MinerU] Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                self.logger.warning(
                    f"[MinerU] Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))

            bottom -= img0.size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    self.logger.warning(
                        f"[MinerU] Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                bottom -= page.size[1]

        if not imgs:
            if need_position:
                return None, None
            return

        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    @staticmethod
    def extract_positions(txt: str):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    # def _show_mineru_raw_blocks(self, content_list, max_blocks=100, page_start=None, page_end=None):
    #     """
    #     MinerU content_list의 각 블록의 type, text_level, text 등 주요 정보를 로그로 출력합니다.
    #     Args:
    #         content_list (list): mineru의 content_list.json에서 읽은 블록 리스트
    #         max_blocks (int): 출력할 최대 블록 수
    #         page_start (int, optional): 출력할 시작 페이지 인덱스 (포함)
    #         page_end (int, optional): 출력할 끝 페이지 인덱스 (포함)
    #     """
    #     def safe_str_for_logging(s, max_length=50):
    #         s = str(s)
    #         if len(s) > max_length:
    #             return s[:max_length] + "..."
    #         return s
        
    #     # 페이지 필터링
    #     if page_start is not None or page_end is not None:
    #         filtered_blocks = []
    #         for idx, block in enumerate(content_list):
    #             page_idx = block.get("page_idx", -1)
    #             if page_start is not None and page_idx < page_start:
    #                 continue
    #             if page_end is not None and page_idx > page_end:
    #                 continue
    #             filtered_blocks.append((idx, block))
    #         target_blocks = filtered_blocks[:max_blocks]
    #         if page_start is not None and page_end is not None:
    #             page_desc = f"페이지 {page_start}~{page_end}의"
    #         elif page_start is not None:
    #             page_desc = f"페이지 {page_start}부터의"
    #         elif page_end is not None:
    #             page_desc = f"페이지 {page_end}까지의"
    #         else:
    #             page_desc = "지정된 페이지 범위의"
    #     else:
    #         target_blocks = list(enumerate(content_list[:max_blocks]))
    #         page_desc = "처음"
        
    #     self.logger.info(f"[MinerU-INFO] ===== {page_desc} {len(target_blocks)}개 블록 RAW 데이터 =====")
    #     for idx, block in target_blocks:
    #         block_type = block.get("type", "unknown")
    #         text_level = block.get("text_level", "없음")
    #         text_preview = safe_str_for_logging(block.get("text", ""), max_length=50)
    #         page_idx = block.get("page_idx", "?")
    #         bbox = block.get("bbox", "없음")
    #         self.logger.info(f"[MinerU-INFO] [블록 #{idx}]")
    #         self.logger.info(f"[MinerU-INFO]   type: {block_type}")
    #         if block_type == "text":
    #             self.logger.info(f"[MinerU-INFO]   text_level: {text_level}")
    #         self.logger.info(f"[MinerU-INFO]   page_idx: {page_idx}")
    #         self.logger.info(f"[MinerU-INFO]   bbox: {bbox}")
    #         self.logger.info(f"[MinerU-INFO]   text_preview: '{text_preview}...'")
    #         self.logger.info(f"[MinerU-INFO]   전체 키: {list(block.keys())}")
    #         # table_caption 내용 출력
    #         if "table_caption" in block:
    #             table_caption = block.get("table_caption", [])
    #             self.logger.info(f"[MinerU-INFO]   table_caption: {table_caption}")
    #     self.logger.info("[MinerU-INFO] ===== RAW 데이터 출력 끝 =====")

    def _show_mineru_raw_blocks(self, content_list, max_blocks=100, page_start=None, page_end=None):
        """
        MinerU content_list의 각 블록의 type, text_level, text 등 주요 정보를 로그로 출력합니다.
        Args:
            content_list (list): mineru의 content_list.json에서 읽은 블록 리스트
            max_blocks (int): 출력할 최대 블록 수
            page_start (int, optional): 출력할 시작 페이지 인덱스 (포함)
            page_end (int, optional): 출력할 끝 페이지 인덱스 (포함)
        """
        def safe_str_for_logging(s, max_length=50):
            s = str(s)
            if len(s) > max_length:
                return s[:max_length] + "..."
            return s
        
        # 페이지 필터링
        if page_start is not None or page_end is not None:
            filtered_blocks = []
            for idx, block in enumerate(content_list):
                page_idx = block.get("page_idx", -1)
                if page_start is not None and page_idx < page_start:
                    continue
                if page_end is not None and page_idx > page_end:
                    continue
                filtered_blocks.append((idx, block))
            target_blocks = filtered_blocks[:max_blocks]
            if page_start is not None and page_end is not None:
                page_desc = f"페이지 {page_start}~{page_end}의"
            elif page_start is not None:
                page_desc = f"페이지 {page_start}부터의"
            elif page_end is not None:
                page_desc = f"페이지 {page_end}까지의"
            else:
                page_desc = "지정된 페이지 범위의"
        else:
            target_blocks = list(enumerate(content_list[:max_blocks]))
            page_desc = "처음"
        
        self.logger.info(f"[MinerU-INFO] ===== {page_desc} {len(target_blocks)}개 블록 RAW 데이터 =====")
        for idx, block in target_blocks:
            block_type = block.get("type", "unknown")
            text_level = block.get("text_level", "없음")
            page_idx = block.get("page_idx", "?")
            bbox = block.get("bbox", "없음")
            
            self.logger.info(f"[MinerU-INFO] [블록 #{idx}]")
            self.logger.info(f"[MinerU-INFO]   type: {block_type}")
            if block_type == "text":
                self.logger.info(f"[MinerU-INFO]   text_level: {text_level}")
            self.logger.info(f"[MinerU-INFO]   page_idx: {page_idx}")
            self.logger.info(f"[MinerU-INFO]   bbox: {bbox}")
            
            # 블록 타입에 따라 다른 미리보기 표시
            if block_type == "table":
                # 테이블인 경우 table_body 정보 표시
                table_body = block.get("table_body", "")
                table_body_len = len(table_body) if table_body else 0
                table_body_preview = safe_str_for_logging(table_body, max_length=100)
                self.logger.info(f"[MinerU-INFO]   table_body_len: {table_body_len}")
                self.logger.info(f"[MinerU-INFO]   table_body_preview: '{table_body_preview}'")
            else:
                # 일반 텍스트 블록
                text_preview = safe_str_for_logging(block.get("text", ""), max_length=50)
                self.logger.info(f"[MinerU-INFO]   text_preview: '{text_preview}'")
            
            self.logger.info(f"[MinerU-INFO]   전체 키: {list(block.keys())}")
            
            # table_caption 내용 출력
            if "table_caption" in block:
                table_caption = block.get("table_caption", [])
                self.logger.info(f"[MinerU-INFO]   table_caption: {table_caption}")
            
            # table_footnote 내용 출력
            if "table_footnote" in block:
                table_footnote = block.get("table_footnote", [])
                self.logger.info(f"[MinerU-INFO]   table_footnote: {table_footnote}")
            
            # img_path 출력 (테이블 이미지 경로)
            if "img_path" in block:
                img_path = block.get("img_path", "")
                self.logger.info(f"[MinerU-INFO]   img_path: {img_path}")
                
        self.logger.info("[MinerU-INFO] ===== RAW 데이터 출력 끝 =====")

    def _preprocess_blocks(self, content_list: list[dict[str, Any]]):
        """
        블록 전처리: discarded 블록과 텍스트 블록을 비교하여 header/footer 후보 감지
        
        처리 로직:
        1. 텍스트 블록 중 discarded 블록과 텍스트 내용이 동일하고 bbox 위치, 높이, 넓이가 10% 범위에 있는 경우 header로 분류
        2. 텍스트 블록 중 같은 페이지의 모든 블록보다 아래에 위치하고 페이지 번호 패턴이 있으면 footer로 분류
        
        Args:
            content_list: MinerU의 content_list (블록 리스트)
        """
        import re
        
        self.logger.info(f"[MinerU-PREPROCESS] ===== 블록 전처리 시작 (Header/Footer 후보 감지) =====")
        
        # 페이지 번호 패턴
        page_number_patterns = [
            re.compile(r'^\d+$'),                    # "1", "23"
            re.compile(r'^-\s*\d+\s*-$'),           # "- 1 -", "-23-"
            re.compile(r'^page\s+\d+', re.I),       # "Page 1", "page 23"
            re.compile(r'^페이지\s+\d+', re.I),      # "페이지 1"
            re.compile(r'^\d+\s*/\s*\d+$'),         # "1 / 10"
        ]
        
        def is_page_number(text):
            """페이지 번호 패턴인지 확인"""
            text = text.strip()
            return any(pattern.match(text) for pattern in page_number_patterns)
        
        def bbox_similar(bbox1, bbox2, tolerance=0.10):
            """두 bbox가 유사한지 확인 (10% 허용 오차)"""
            if not bbox1 or not bbox2:
                return False
            if len(bbox1) != 4 or len(bbox2) != 4:
                return False
            
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            width1 = x2_1 - x1_1
            height1 = y2_1 - y1_1
            width2 = x2_2 - x1_2
            height2 = y2_2 - y1_2
            
            # 위치 비교 (중심점 기준)
            center_x1 = (x1_1 + x2_1) / 2
            center_y1 = (y1_1 + y2_1) / 2
            center_x2 = (x1_2 + x2_2) / 2
            center_y2 = (y1_2 + y2_2) / 2
            
            # 10% 허용 오차 (정규화 범위 1000 기준)
            position_tolerance = 1000 * tolerance
            if abs(center_x1 - center_x2) > position_tolerance:
                return False
            if abs(center_y1 - center_y2) > position_tolerance:
                return False
            
            # 크기 비교 (상대적 10% 허용)
            if width1 > 0 and abs(width2 - width1) / width1 > tolerance:
                return False
            if height1 > 0 and abs(height2 - height1) / height1 > tolerance:
                return False
            
            return True
        
        def text_similar(text1, text2):
            """두 텍스트가 동일한지 확인 (공백 무시)"""
            return text1.strip() == text2.strip()
        
        # 페이지별 블록 맵 생성
        page_blocks_map = {}
        for idx, chunk in enumerate(content_list):
            page_idx = chunk.get("page_idx", -1)
            if page_idx not in page_blocks_map:
                page_blocks_map[page_idx] = []
            page_blocks_map[page_idx].append(idx)
        
        # discarded 블록 수집 및 별표 타이틀 블록 변환
        discarded_blocks = []
        # "별표" 텍스트가 포함되면 별표 블록으로 인식 (이후 텍스트는 무시)
        # "별표 N", "별표N", "[별표 1]", "[별표1]", "【별표 1】", "【 별표 7 】 <개정 2023.07.07.>" 등 모든 변형 매칭
        byulpyo_pattern = re.compile(r'[\[\【<(]?\s*별표\s*', re.IGNORECASE)

        for idx, chunk in enumerate(content_list):
            if chunk.get("type", "").lower() == "discarded":
                text = chunk.get("text", "").strip()
                page_idx = chunk.get("page_idx", -1)

                # 별표 패턴 검사 (별표는 기존 처리 유지)
                if byulpyo_pattern.match(text):
                    # discarded → text 블록으로 변환 (text_level=1로 타이틀 처리)
                    chunk["type"] = "text"
                    chunk["text_level"] = 1
                    chunk["_converted_from"] = "discarded_byulpyo"
                    self.logger.info(
                        f"[MinerU-PREPROCESS] 별표 블록 변환 | idx={idx} | page_idx={page_idx} | text={text[:50]} → text (level=1)"
                    )
                else:
                    discarded_blocks.append({
                        "idx": idx,
                        "text": text,
                        "bbox": chunk.get("bbox"),
                        "page_idx": page_idx
                    })
                    # 디버깅용 로그 출력
                    self.logger.info(
                        f"[MinerU-PREPROCESS] Discarded 블록 | idx={idx} | page_idx={page_idx} | bbox={chunk.get('bbox')} | text={text[:50]}"
                    )

        # discarded 블록 중복 분석 및 처리
        header_footer_texts = set()  # header/footer 원본 텍스트 목록 (완전한 텍스트 저장)
        if discarded_blocks:
            # 텍스트별 빈도 계산 (첫 10글자를 키로 사용하되 원본 텍스트도 함께 저장)
            text_counts = {}
            for block in discarded_blocks:
                original_text = block["text"].strip()
                # 첫 10글자를 키로 사용
                text_key = original_text[:10]
                
                if text_key not in text_counts:
                    text_counts[text_key] = []
                text_counts[text_key].append(block)

            # 2개 이상 중복되는 것은 header/footer로 유지, 1개만 있는 것은 text로 변경
            for text_key, blocks in text_counts.items():
                if len(blocks) >= 2:
                    # 2개 이상 중복: header/footer로 유지 (discarded 유지)
                    # 원본 텍스트들을 모두 header_footer_texts에 추가
                    for block in blocks:
                        original_text = block["text"].strip()
                        header_footer_texts.add(original_text)
                    
                    self.logger.info(f"[MinerU-PREPROCESS] 중복 discarded 유지 (header/footer) | 키='{text_key}' | 출현={len(blocks)}회 | 원본 텍스트 {len(blocks)}개 저장")
                else:
                    # 1개만 있는 경우: 일반 텍스트 블록으로 변경
                    block = blocks[0]
                    idx = block["idx"]
                    content_list[idx]["type"] = "text"
                    content_list[idx]["text_level"] = 0  # 일반 텍스트
                    content_list[idx]["_converted_from"] = "discarded_unique"
                    self.logger.info(f"[MinerU-PREPROCESS] 고유 discarded → text 변환 | idx={idx} | 텍스트='{text_key[:50]}...'")

        # 각 페이지의 첫/마지막 텍스트 블록에서 header/footer 텍스트 제거
        if header_footer_texts:
            self.logger.info(f"[MinerU-PREPROCESS] ===== 각 페이지 첫/마지막 텍스트 블록에서 header/footer 제거 =====")

            for page_idx, block_indices in page_blocks_map.items():
                # 해당 페이지의 텍스트 블록만 필터링 (text_level=0)
                text_blocks_in_page = []
                for idx in block_indices:
                    chunk = content_list[idx]
                    if chunk.get("type") == "text" and chunk.get("text_level", 0) == 0:
                        text_blocks_in_page.append(idx)

                if not text_blocks_in_page:
                    continue

                # 첫 번째와 마지막 3개 텍스트 블록
                first_text_idx = text_blocks_in_page[0]
                # 마지막 3개 텍스트 블록 (페이지에 3개 미만이면 있는 만큼만)
                last_text_indices = text_blocks_in_page[-3:] if len(text_blocks_in_page) >= 3 else text_blocks_in_page[-1:]

                # 첫 번째 텍스트 블록에서 header/footer 제거
                first_chunk = content_list[first_text_idx]
                first_text = first_chunk.get("text", "")
                original_first_text = first_text

                for header_text in header_footer_texts:
                    if header_text in first_text:
                        first_text = first_text.replace(header_text, "").strip()
                        self.logger.info(f"[MinerU-PREPROCESS] 첫 텍스트 블록 header 제거 | 페이지={page_idx} | idx={first_text_idx} | 제거='{header_text[:30]}...'")

                if first_text != original_first_text:
                    content_list[first_text_idx]["text"] = first_text
                    self.logger.info(f"[MinerU-PREPROCESS] 첫 텍스트 블록 정리 완료 | idx={first_text_idx} | 결과='{first_text[:80]}...'")

                # 마지막 3개 텍스트 블록에서 header/footer 제거
                for last_text_idx in last_text_indices:
                    last_chunk = content_list[last_text_idx]
                    last_text = last_chunk.get("text", "")
                    original_last_text = last_text

                    for footer_text in header_footer_texts:
                        if footer_text in last_text:
                            last_text = last_text.replace(footer_text, "").strip()
                            self.logger.info(f"[MinerU-PREPROCESS] 마지막 텍스트 블록 footer 제거 | 페이지={page_idx} | idx={last_text_idx} | 제거='{footer_text[:30]}...'")

                    if last_text != original_last_text:
                        content_list[last_text_idx]["text"] = last_text
                        self.logger.info(f"[MinerU-PREPROCESS] 마지막 텍스트 블록 정리 완료 | idx={last_text_idx} | 결과='{last_text[:80]}...'")

            self.logger.info(f"[MinerU-PREPROCESS] ===== header/footer 제거 완료 =====")

        self.logger.info(f"[MinerU-PREPROCESS] Discarded 블록 수: {len(discarded_blocks)}")
        
        # 1. 텍스트 블록 중 discarded 블록과 유사한 것 찾기 (header 후보)
        header_candidates = []
        for idx, chunk in enumerate(content_list):
            if chunk.get("type", "") == "text":
                # 이미 discarded에서 text로 변환된 블록은 다시 discarded로 분류하지 않음
                if chunk.get("_converted_from") == "discarded_unique":
                    continue
                    
                text = chunk.get("text", "").strip()
                bbox = chunk.get("bbox")
                page_idx = chunk.get("page_idx", -1)
                
                for discarded in discarded_blocks:
                    if text_similar(text, discarded["text"]) and bbox_similar(bbox, discarded["bbox"]):
                        header_candidates.append({
                            "idx": idx,
                            "text": text,
                            "bbox": bbox,
                            "page_idx": page_idx,
                            "matched_discarded_idx": discarded["idx"],
                            "reason": "discarded와 텍스트/위치 일치"
                        })
                        break
        
        # 2. 텍스트 블록 중 페이지 하단에 위치하고 페이지 번호 패턴인 것 찾기 (footer 후보)
        footer_candidates = []
        for page_idx, block_indices in page_blocks_map.items():
            if not block_indices:
                continue
            
            # 해당 페이지의 모든 블록 bbox 수집
            page_bboxes = []
            for idx in block_indices:
                bbox = content_list[idx].get("bbox")
                if bbox and len(bbox) == 4:
                    page_bboxes.append(bbox)
            
            if not page_bboxes:
                continue
            
            # 페이지의 최하단 y 좌표
            max_y = max(bbox[3] for bbox in page_bboxes)  # bbox[3] = y2 (하단)
            
            # 텍스트 블록 검사
            for idx in block_indices:
                chunk = content_list[idx]
                if chunk.get("type", "") == "text":
                    text = chunk.get("text", "").strip()
                    bbox = chunk.get("bbox")
                    
                    if bbox and len(bbox) == 4:
                        y2 = bbox[3]
                        # 페이지 최하단에 위치 (10% 허용)
                        if y2 >= max_y * 0.90 and is_page_number(text):
                            footer_candidates.append({
                                "idx": idx,
                                "text": text,
                                "bbox": bbox,
                                "page_idx": page_idx,
                                "reason": "페이지 하단 + 페이지 번호 패턴"
                            })
        
        # 결과 출력
        self.logger.info(f"[MinerU-PREPROCESS] ===== Header 후보 ({len(header_candidates)}개) =====")
        for candidate in header_candidates:
            idx = candidate["idx"]
            text = candidate["text"][:50]
            bbox = candidate["bbox"]
            page_idx = candidate["page_idx"]
            reason = candidate["reason"]
            self.logger.info(f"[MinerU-PREPROCESS]   [블록 #{idx}] Page {page_idx}, bbox={bbox}")
            self.logger.info(f"[MinerU-PREPROCESS]     텍스트: '{text}...'")
            self.logger.info(f"[MinerU-PREPROCESS]     이유: {reason}")
        
        self.logger.info(f"[MinerU-PREPROCESS] ===== Footer 후보 ({len(footer_candidates)}개) =====")
        for candidate in footer_candidates:
            idx = candidate["idx"]
            text = candidate["text"][:50]
            bbox = candidate["bbox"]
            page_idx = candidate["page_idx"]
            reason = candidate["reason"]
            self.logger.info(f"[MinerU-PREPROCESS]   [블록 #{idx}] Page {page_idx}, bbox={bbox}")
            self.logger.info(f"[MinerU-PREPROCESS]     텍스트: '{text}...'")
            self.logger.info(f"[MinerU-PREPROCESS]     이유: {reason}")
        
        # header/footer 후보를 discarded로 표시
        header_footer_indices = set()
        for candidate in header_candidates:
            idx = candidate["idx"]
            content_list[idx]["type"] = "discarded"
            content_list[idx]["_discarded_reason"] = candidate["reason"]
            header_footer_indices.add(idx)
            self.logger.info(f"[MinerU-PREPROCESS] 블록 #{idx}를 discarded로 표시 (Header)")
        
        for candidate in footer_candidates:
            idx = candidate["idx"]
            content_list[idx]["type"] = "discarded"
            content_list[idx]["_discarded_reason"] = candidate["reason"]
            header_footer_indices.add(idx)
            self.logger.info(f"[MinerU-PREPROCESS] 블록 #{idx}를 discarded로 표시 (Footer)")
        
        # ===== 별표 블록 처리: 별표와 테이블 사이의 텍스트를 caption으로 할당 =====
        self.logger.info(f"[MinerU-PREPROCESS] ===== 별표 블록 처리 시작 =====")
        # "별표" 텍스트가 포함되면 별표 블록으로 인식 (이후 텍스트는 무시)
        # "별표 N", "별표N", "[별표 1]", "[별표1]", "【별표 1】", "【 별표 7 】 <개정 2023.07.07.>" 등 모든 변형 매칭
        byulpyo_pattern = re.compile(r'[\[\【<(]?\s*별표\s*', re.IGNORECASE)

        # 페이지별로 블록 그룹화
        page_blocks_map = {}
        for idx, chunk in enumerate(content_list):
            page_idx = chunk.get("page_idx", -1)
            if page_idx not in page_blocks_map:
                page_blocks_map[page_idx] = []
            page_blocks_map[page_idx].append(idx)

        # 각 페이지별로 별표 블록 처리
        for page_idx, block_indices in page_blocks_map.items():
            if not block_indices:
                continue

            # 해당 페이지의 별표 블록 찾기
            byulpyo_blocks_in_page = []
            for idx in block_indices:
                chunk = content_list[idx]
                chunk_type = chunk.get("type", "").lower()
                text = chunk.get("text", "").strip()

                if chunk_type == "text" and byulpyo_pattern.match(text):
                    byulpyo_blocks_in_page.append({
                        "idx": idx,
                        "text": text,
                        "page_idx": page_idx
                    })

            # 페이지 내 별표 블록들 처리 (bbox y 좌표 기준으로 정렬)
            if byulpyo_blocks_in_page:
                # bbox y 좌표로 정렬 (페이지 상단부터 하단으로)
                byulpyo_blocks_in_page.sort(key=lambda x: content_list[x["idx"]].get("bbox", [0, 0, 0, 0])[1])
                
                for byulpyo_info in byulpyo_blocks_in_page:
                    byulpyo_idx = byulpyo_info["idx"]
                    byulpyo_text = byulpyo_info["text"]
                    byulpyo_bbox = content_list[byulpyo_idx].get("bbox", [0, 0, 0, 0])

                    self.logger.info(f"[MinerU-PREPROCESS] 별표 블록 발견 | idx={byulpyo_idx} | page_idx={page_idx} | bbox={byulpyo_bbox} | text={byulpyo_text[:50]}")

                    # 디버깅: 페이지 내 모든 블록들 출력
                    self.logger.info(f"[MinerU-PREPROCESS]   → 페이지 {page_idx} 내 모든 블록들:")
                    for block_order, block_idx in enumerate(block_indices):
                        block = content_list[block_idx]
                        block_type = block.get("type", "")
                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                        block_text = block.get("text", "")[:30]
                        
                        if block_type == "table":
                            # 테이블 속성 추가 출력 (MinerU 원본은 table_caption을 사용)
                            table_caption = block.get("table_caption", [])
                            table_body = block.get("table_body", "")
                            table_footnote = block.get("table_footnote", [])
                            self.logger.info(f"[MinerU-PREPROCESS]     [{block_order}] 블록 #{block_idx} | page={page_idx} | bbox={block_bbox} | type={block_type} | text='{block_text}'")
                            self.logger.info(f"[MinerU-PREPROCESS]       테이블 속성: table_caption={table_caption} | table_body='{table_body[:50]}...' | table_footnote={table_footnote}")
                        else:
                            self.logger.info(f"[MinerU-PREPROCESS]     [{block_order}] 블록 #{block_idx} | page={page_idx} | bbox={block_bbox} | type={block_type} | text='{block_text}'")

                    # 같은 페이지에서 별표 블록 bbox보다 아래에 있는 블록들 찾기
                    caption_parts = []
                    table_idx = None

                    # 페이지 내에서 bbox y 좌표로 정렬된 블록들 순회
                    sorted_blocks = sorted(block_indices, key=lambda idx: content_list[idx].get("bbox", [0, 0, 0, 0])[1])

                    for idx in sorted_blocks:
                        if idx == byulpyo_idx:
                            continue  # 별표 블록 자신은 건너뜀

                        next_chunk = content_list[idx]
                        next_bbox = next_chunk.get("bbox", [0, 0, 0, 0])

                        # 별표 블록보다 아래에 있는 블록만 처리
                        if next_bbox[1] <= byulpyo_bbox[1]:
                            continue  # 위에 있는 블록은 건너뜀

                        next_type = next_chunk.get("type", "").lower()
                        next_text = next_chunk.get("text", "").strip()

                        # 테이블 또는 이미지 발견 시 처리
                        if next_type == "table":
                            table_idx = idx
                            self.logger.info(f"[MinerU-PREPROCESS]   → 테이블 발견 | idx={table_idx} | bbox={next_bbox}")
                            break
                        elif next_type == "image":
                            image_idx = idx
                            self.logger.info(f"[MinerU-PREPROCESS]   → 이미지 발견 | idx={image_idx} | bbox={next_bbox}")
                            break

                        # text 블록이면 caption에 추가
                        if next_type == "text" and next_text:
                            # 다른 별표 블록을 만나면 중단
                            if byulpyo_pattern.match(next_text):
                                self.logger.info(f"[MinerU-PREPROCESS]   → 다른 별표 블록 만남 | idx={idx}, 중단")
                                break

                            caption_parts.append(next_text)
                            self.logger.info(f"[MinerU-PREPROCESS]   → Caption 텍스트 수집 | idx={idx} | bbox={next_bbox} | text={next_text[:50]}")

                    # 테이블이 발견되면 caption 할당
                    if table_idx is not None:
                        # 별표 텍스트 + 수집된 텍스트를 caption으로 설정
                        full_caption = byulpyo_text
                        if caption_parts:
                            full_caption += "\n" + "\n".join(caption_parts)

                        # 기존 table_caption 확인 (MinerU 원본은 table_caption을 리스트로 사용)
                        existing_table_caption = content_list[table_idx].get("table_caption", [])
                        existing_caption_str = "\n".join(existing_table_caption) if existing_table_caption else ""
                        
                        self.logger.info(f"[MinerU-PREPROCESS]   → Caption 구성 요소:")
                        self.logger.info(f"[MinerU-PREPROCESS]     byulpyo_text: '{byulpyo_text}'")
                        self.logger.info(f"[MinerU-PREPROCESS]     caption_parts: {caption_parts}")
                        self.logger.info(f"[MinerU-PREPROCESS]     full_caption: '{full_caption}'")
                        self.logger.info(f"[MinerU-PREPROCESS]     existing_table_caption: {existing_table_caption}")

                        # caption 속성에 별표 캡션 설정 (별도 속성으로 관리)
                        # caption_parts가 없어도 기존 table_caption과 합쳐서 caption으로 설정
                        if existing_caption_str:
                            content_list[table_idx]["caption"] = full_caption + "\n" + existing_caption_str
                            self.logger.info(f"[MinerU-PREPROCESS]     최종 caption (기존 table_caption 뒤에 추가): '{content_list[table_idx]['caption']}'")
                        else:
                            content_list[table_idx]["caption"] = full_caption
                            self.logger.info(f"[MinerU-PREPROCESS]     최종 caption (새로 설정): '{content_list[table_idx]['caption']}'")

                        self.logger.info(f"[MinerU-PREPROCESS]   ✓ Caption 할당 완료 | 테이블 idx={table_idx}")
                        self.logger.info(f"[MinerU-PREPROCESS]     Caption: {full_caption[:100]}...")

                        # 별표 블록과 caption으로 사용된 텍스트 블록들을 discarded로 표시
                        content_list[byulpyo_idx]["type"] = "discarded"
                        content_list[byulpyo_idx]["_discarded_reason"] = "별표 블록 (테이블 caption으로 사용됨)"

                        for idx in sorted_blocks:
                            chunk = content_list[idx]
                            chunk_bbox = chunk.get("bbox", [0, 0, 0, 0])
                            # 별표 블록보다 아래에 있고, 테이블보다 위에 있는 텍스트 블록들
                            if (chunk_bbox[1] > byulpyo_bbox[1] and 
                                idx < table_idx and 
                                chunk.get("type", "").lower() == "text"):
                                chunk["type"] = "discarded"
                                chunk["_discarded_reason"] = "별표-테이블 사이 텍스트 (caption으로 사용됨)"
                                self.logger.info(f"[MinerU-PREPROCESS]     블록 #{idx}를 discarded로 표시")

                    # 같은 페이지에 테이블이 없으면 다음 페이지의 첫 번째 테이블에 할당 시도
                    elif caption_parts and not table_idx:
                        self.logger.info(f"[MinerU-PREPROCESS]   → 같은 페이지에 테이블 없음, 바로 다음 페이지 검색 시작")

                        # 바로 다음 페이지만 체크
                        next_page_idx = page_idx + 1
                        next_page_found = False

                        if next_page_idx in page_blocks_map:
                            next_page_blocks = page_blocks_map[next_page_idx]
                            if next_page_blocks:
                                # 다음 페이지의 블록들을 bbox y 좌표로 정렬
                                next_sorted_blocks = sorted(next_page_blocks, key=lambda idx: content_list[idx].get("bbox", [0, 0, 0, 0])[1])

                                # 다음 페이지의 첫 번째 테이블 찾기
                                for idx in next_sorted_blocks:
                                    chunk = content_list[idx]
                                    chunk_type = chunk.get("type", "").lower()

                                    if chunk_type == "table":
                                        table_idx = idx
                                        table_bbox = chunk.get("bbox", [0, 0, 0, 0])

                                        # 별표 텍스트 + 수집된 텍스트를 caption으로 설정
                                        full_caption = byulpyo_text
                                        if caption_parts:
                                            full_caption += "\n" + "\n".join(caption_parts)

                                        # 기존 table_caption 확인 (MinerU 원본은 table_caption을 리스트로 사용)
                                        existing_table_caption = content_list[table_idx].get("table_caption", [])
                                        existing_caption_str = "\n".join(existing_table_caption) if existing_table_caption else ""
                                        
                                        self.logger.info(f"[MinerU-PREPROCESS]   → 다음 페이지 Caption 구성 요소:")
                                        self.logger.info(f"[MinerU-PREPROCESS]     byulpyo_text: '{byulpyo_text}'")
                                        self.logger.info(f"[MinerU-PREPROCESS]     caption_parts: {caption_parts}")
                                        self.logger.info(f"[MinerU-PREPROCESS]     full_caption: '{full_caption}'")
                                        self.logger.info(f"[MinerU-PREPROCESS]     existing_table_caption: {existing_table_caption}")

                                        # caption 속성에 별표 캡션 설정 (별도 속성으로 관리)
                                        # 기존 table_caption이 있으면 뒤에 추가
                                        if existing_caption_str:
                                            content_list[table_idx]["caption"] = full_caption + "\n" + existing_caption_str
                                            self.logger.info(f"[MinerU-PREPROCESS]     최종 caption (기존 table_caption 뒤에 추가): '{content_list[table_idx]['caption']}'")
                                        else:
                                            content_list[table_idx]["caption"] = full_caption
                                            self.logger.info(f"[MinerU-PREPROCESS]     최종 caption (새로 설정): '{content_list[table_idx]['caption']}'")

                                        self.logger.info(f"[MinerU-PREPROCESS]   ✓ 다음 페이지 테이블에 Caption 할당 | 페이지={next_page_idx} | 테이블 idx={table_idx} | bbox={table_bbox}")
                                        self.logger.info(f"[MinerU-PREPROCESS]     Caption: {full_caption[:100]}...")

                                        # 별표 블록을 discarded로 표시
                                        content_list[byulpyo_idx]["type"] = "discarded"
                                        content_list[byulpyo_idx]["_discarded_reason"] = "별표 블록 (다음 페이지 테이블 caption으로 사용됨)"

                                        next_page_found = True
                                        break

                        if not next_page_found:
                            self.logger.info(f"[MinerU-PREPROCESS]   → 바로 다음 페이지({next_page_idx})에 테이블 없음, caption 할당 실패")
                            self.logger.info(f"[MinerU-PREPROCESS]   → 별표 블록을 일반 텍스트 블록으로 유지 | idx={byulpyo_idx}")

        self.logger.info(f"[MinerU-PREPROCESS] ===== 블록 전처리 끝 =====")
        
        return header_candidates, footer_candidates

    def _custom_merge(self, content_list: list[dict[str, Any]], max_tokens=512):
        """
        조항별로 text 블록을 병합하는 함수
        
        병합 규칙:
        1. text 블록(text_level==0)만 병합 대상
        2. 타이틀 블록(text_level>0), 테이블, 이미지 등은 병합하지 않음
        3. article_pattern, number_pattern, chapter_pattern으로 병합
        4. 병합된 블록의 bbox를 업데이트
        5. 페이지가 변경되면 첫 블록의 bbox로 업데이트
        6. 토큰 수 제한: max_tokens 초과 시 병합 중단
        
        Args:
            content_list: MinerU의 content_list (블록 리스트)
            max_tokens: 최대 토큰 수 제한 (기본 512)
        
        Returns:
            merged_content_list: 병합된 블록 리스트
        """
        from common.token_utils import num_tokens_from_string
        
        self.logger.info(f"[MinerU-MERGE] ===== 커스텀 병합 시작: {len(content_list)} blocks (max_tokens={max_tokens}) =====")
        
        # 0단계: 타이틀 블록 재분류 (1단계 병합 전에 수행)
        content_list = self._reclassify_title_blocks(content_list, enable_reclassify=True)
        
        # 조항/장 패턴
        article_pattern = re.compile(r'^제\s*(\d+(?:-\d+)?)\s*조(?:의\s*\d+)?')
        chapter_pattern = re.compile(r'^제\s*(\d+)\s*장')
        
        merged_content_list = []
        i = 0
        
        while i < len(content_list):
            block = content_list[i]
            block_type = block.get("type", "").lower()
            text_level = block.get("text_level", 0)
            text = block.get("text", "")
            
            # text 블록이 아니면 그대로 추가
            if block_type != "text":
                merged_content_list.append(block)
                i += 1
                continue
            
            # 패턴 매칭
            match = article_pattern.match(text.strip())
            
            # 패턴이 없으면 그대로 추가
            # (text_level > 0이지만 패턴 없는 title 블록 포함)
            if not match:
                merged_content_list.append(block)
                i += 1
                continue
            
            # 패턴 발견 - 병합 시작
            article_num = match.group(1)  # string으로 저장
            current_article_id = match.group(0)
            self.logger.info(f"[MinerU-MERGE] {current_article_id} 병합 시작 (블록 {i})")
            
            # 병합 시작
            merged_text = text
            merged_text_list = [text]  # 병합된 텍스트 리스트를 루프 바깥에서 선언/초기화
            merged_page_idx = block.get("page_idx", 0)
            merged_bbox = block.get("bbox", [0, 0, 0, 0])
            merged_token_count = num_tokens_from_string(merged_text)  # 토큰 수 초기화
            
            # bbox 유효성 검증 강화
            if merged_bbox and isinstance(merged_bbox, list) and len(merged_bbox) == 4 and merged_bbox != [0, 0, 0, 0]:
                merged_bbox = merged_bbox.copy()
                min_x1, min_y1, max_x2, max_y2 = merged_bbox
                has_valid_bbox = True
            else:
                min_x1 = min_y1 = max_x2 = max_y2 = 0
                has_valid_bbox = False
                # bbox가 없으면 기본값 설정
                merged_bbox = [0, 0, 0, 0]
            
            first_page_idx = merged_page_idx
            
            # 다음 블록들 확인
            j = i + 1
            
            while j < len(content_list):
                next_block = content_list[j]
                next_block_type = next_block.get("type", "").lower()
                next_text_level = next_block.get("text_level", 0)
                next_text = next_block.get("text", "")
                next_page_idx = next_block.get("page_idx", 0)
                
                # discarded 블록은 건너뛰고 계속 진행
                if next_block_type == "discarded":
                    self.logger.info(
                        f"[MinerU-MERGE] 블록 {j}는 discarded, 건너뛰고 병합 계속 | text={next_text.strip()[:30]}"
                    )
                    j += 1
                    continue
                
                # text 블록이 아니면 중단 (table, image, equation 등)
                if next_block_type != "text":
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}는 {next_block_type}, 병합 중단")
                    break
                
                # 다음 조항/장 패턴 확인 (타이틀 블록이어도 패턴 체크)
                next_match = article_pattern.match(next_text.strip())
                chapter_match = chapter_pattern.match(next_text.strip())
                
                # 장 패턴이면 중단
                if chapter_match:
                    chapter_num = chapter_match.group(1)
                    self.logger.info(f"[MinerU-MERGE] 제{chapter_num}장 발견, 병합 중단")
                    break
                
                # 다음 조항 패턴이 발견되면 무조건 중단
                if next_match:
                    next_article_id = next_match.group(0)
                    self.logger.info(f"[MinerU-MERGE] 조항 패턴 {next_article_id} 발견 (현재: {current_article_id}), 병합 중단")
                    break
                
                # 패턴 없는 타이틀 블록(text_level > 0)이면 병합 중단
                # (예: "제1절", "제목 텍스트" 등)
                if next_text_level > 0:
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}는 패턴 없는 title (level {next_text_level}), 병합 중단")
                    break
                
                # 토큰 수 체크 (병합 전)
                next_token_count = num_tokens_from_string(next_text)
                if merged_token_count + next_token_count > max_tokens:
                    self.logger.info(f"[MinerU-MERGE] 토큰 수 초과 ({merged_token_count + next_token_count} > {max_tokens}), 병합 중단")
                    break
                
                # 병합 수행 (패턴도 없고 타이틀도 아닌 일반 텍스트)
                merged_text += "\n" + next_text
                merged_text_list.append(next_text)
                merged_token_count += next_token_count  # 토큰 수 누적
                
                # bbox 업데이트
                next_bbox = next_block.get("bbox")
                if has_valid_bbox and next_bbox and len(next_bbox) == 4:
                    # 페이지가 변경되면 첫 블록의 bbox로 유지
                    if next_page_idx == first_page_idx:
                        # 같은 페이지면 bbox 확장
                        x1, y1, x2, y2 = next_bbox
                        min_x1 = min(min_x1, x1)
                        min_y1 = min(min_y1, y1)
                        max_x2 = max(max_x2, x2)
                        max_y2 = max(max_y2, y2)
                    # 페이지가 변경되면 첫 블록의 bbox 유지 (업데이트 안 함)
                
                j += 1
            
            # 병합된 블록 생성
            if has_valid_bbox:
                merged_bbox = [min_x1, min_y1, max_x2, max_y2]
            else:
                # bbox가 유효하지 않으면 기본값 보장
                merged_bbox = [0, 0, 0, 0]
            
            merged_block = {
                "type": "text",
                "text": merged_text,
                "text_level": 0,
                "bbox": merged_bbox,
                "page_idx": first_page_idx
            }
            
            # 원본 블록의 다른 속성 복사
            for key in block:
                if key not in ["text", "type", "text_level", "bbox", "page_idx"]:
                    merged_block[key] = block[key]
            
            merged_content_list.append(merged_block)
            self.logger.info(f"[MinerU-MERGE] 블록 {i}~{j-1} 병합 완료 ({j-i}개 블록, {merged_token_count} 토큰)")
            
            i = j
        
        self.logger.info(f"[MinerU-MERGE] 1단계 완료 (패턴 기반): {len(content_list)} -> {len(merged_content_list)} blocks")
        
        # 2단계: 패턴 없는 연속된 text 블록들을 토큰 수 기준으로 병합
        final_merged = self._merge_non_pattern_blocks(merged_content_list, max_tokens=max_tokens)
        
        self.logger.info(f"[MinerU-MERGE] ===== 커스텀 병합 완료: {len(content_list)} -> {len(final_merged)} blocks =====")
        return final_merged

    def _reclassify_title_blocks(self, content_list: list[dict[str, Any]], enable_reclassify=True):
        """
        타이틀 블록 재분류: 진짜 타이틀인지 판단하여 text_level 조정
        
        판단 기준:
        - 제N장, 제N절 같은 패턴이 있으면 타이틀로 유지 (text_level > 0)
        - 그 외는 일반 텍스트로 다운그레이드 (text_level = 0)
        
        Args:
            content_list: 병합 완료된 블록 리스트
            enable_reclassify: 재분류 기능 활성화 여부 (기본 True)
        
        Returns:
            reclassified_list: 재분류된 블록 리스트
        """
        if not enable_reclassify:
            return content_list
        
        self.logger.info(f"[MinerU-RECLASSIFY] ===== 타이틀 블록 재분류 시작 =====")
        
        # 진짜 타이틀 패턴 (제N장, 제N절 등)
        true_title_patterns = [
            re.compile(r'^제\s*\d+\s*장'),  # 제1장, 제2장
            re.compile(r'^제\s*\d+\s*절'),  # 제1절, 제2절
            re.compile(r'^제\s*\d+\s*편'),  # 제1편, 제2편
            re.compile(r'^제\s*\d+\s*부'),  # 제1부, 제2부
            re.compile(r'^제\s*\d+\s*관'),  # 제1관, 제2관
        ]
        
        def is_true_title(text):
            """진짜 타이틀 패턴인지 확인"""
            text = text.strip()
            return any(pattern.match(text) for pattern in true_title_patterns)
        
        def is_centered(center_x, tolerance=10):
            """center_x가 500 ±10 범위에 있는지 확인"""
            return 490 <= center_x <= 510
        
        reclassified_count = 0
        for block in content_list:
            if block.get("type", "").lower() == "text" and block.get("text_level", 0) > 0:
                text = block.get("text", "")
                bbox = block.get("bbox", [])
                
                # bbox 정보 계산
                has_valid_bbox = False
                center_x = 0
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    has_valid_bbox = True
                    bbox_info = f"center_x={center_x:.1f}, width={width:.1f}, height={height:.1f}"
                else:
                    bbox_info = "bbox 없음"
                
                # 모든 text_level > 0 블록 정보 출력
                text_len = len(text)
                self.logger.info(
                    f"[MinerU-RECLASSIFY] 타이틀 블록 분석 | level={block.get('text_level')} | {bbox_info} | len={text_len} | text={text[:50]}"
                )
                
                # 타이틀 판단 기준
                is_pattern_title = is_true_title(text)
                is_centered_title = has_valid_bbox and is_centered(center_x)
                
                # 1. height > 25 and len(text) < 40 이면 title로 분류
                is_large_short_text = has_valid_bbox and height > 25 and text_len < 40
                
                # 2. 중앙 정렬인데 len(text) > 80 이면 title이 아님으로 분류
                is_long_centered_text = is_centered_title and text_len > 80
                
                # 최종 타이틀 판단
                is_title = (is_pattern_title or is_centered_title or is_large_short_text) and not is_long_centered_text
                
                if not is_title:
                    # 진짜 타이틀이 아니면 text_level을 0으로 다운그레이드
                    old_level = block["text_level"]
                    block["text_level"] = 0
                    reclassified_count += 1
                    self.logger.info(
                        f"[MinerU-RECLASSIFY] -> 타이틀 다운그레이드 (level {old_level} -> 0): "
                        f"패턴={is_pattern_title}, 중앙정렬={is_centered_title}, "
                        f"큰글짜짧은텍스트={is_large_short_text}, 긴중앙텍스트={is_long_centered_text}"
                    )
                else:
                    self.logger.info(
                        f"[MinerU-RECLASSIFY] -> 타이틀 유지: "
                        f"패턴={is_pattern_title}, 중앙정렬={is_centered_title}, "
                        f"큰글짜짧은텍스트={is_large_short_text}, 긴중앙텍스트={is_long_centered_text}"
                    )
        
        self.logger.info(
            f"[MinerU-RECLASSIFY] ===== 재분류 완료: {reclassified_count}개 블록 다운그레이드 ====="
        )
        return content_list

    def _merge_non_pattern_blocks(self, content_list: list[dict[str, Any]], max_tokens=512):
        """
        패턴 없는 연속된 text 블록들을 토큰 수 기준으로 병합
        
        병합 대상:
        - text 블록 (text_level 상관없이)
        - 조항/번호/장 패턴이 없는 블록
        
        병합 중단 조건:
        - text 블록이 아닌 경우 (table, image, equation 등)
        - 조항/번호/장 패턴이 있는 경우
        - text_level > 0인 타이틀 블록을 만난 경우
        - 토큰 수가 max_tokens를 초과하는 경우
        
        Args:
            content_list: 1단계 병합 완료된 블록 리스트
            max_tokens: 최대 토큰 수 (기본 512)
        
        Returns:
            final_merged: 최종 병합된 블록 리스트
        """
        from common.token_utils import num_tokens_from_string
        
        self.logger.info(f"[MinerU-MERGE] 2단계 시작 (패턴 없는 블록 병합): {len(content_list)} blocks")
        
        # 조항/번호/장 패턴
        article_pattern = re.compile(r'^제\s*(\d+(?:-\d+)?)\s*조(?:\s*-\s*\d+|\s*의\s*\d+)?')
        number_pattern = re.compile(r'^(\d+(?:[.-]\d+)+)\.?(?:\s|(?=[가-힣]))')
        chapter_pattern = re.compile(r'^제\s*(\d+)\s*장')
        
        def has_pattern(text):
            """텍스트가 조항/번호 패턴을 가지고 있는지 확인"""
            text = text.strip()
            if article_pattern.match(text):
                return True
            if number_pattern.match(text):
                return True
            if chapter_pattern.match(text):
                return True
            return False
        
        final_merged = []
        i = 0
        
        while i < len(content_list):
            block = content_list[i]
            block_type = block.get("type", "").lower()
            text_level = block.get("text_level", 0)
            text = block.get("text", "")
            
            # text 블록이고, 패턴이 없는 경우만 병합 대상 (text_level 상관없이)
            is_mergeable = (
                block_type == "text" and 
                not has_pattern(text)
            )
            
            if not is_mergeable:
                final_merged.append(block)
                i += 1
                continue
            
            # 연속된 패턴 없는 블록들 병합
            self.logger.info(f"[MinerU-MERGE] 패턴 없는 블록 병합 시작 (블록 {i})")
            
            merged_text = text
            merged_text_list = [text]
            merged_page_idx = block.get("page_idx", 0)
            merged_bbox = block.get("bbox", [0, 0, 0, 0])
            
            # bbox 유효성 검증 강화
            if merged_bbox and isinstance(merged_bbox, list) and len(merged_bbox) == 4 and merged_bbox != [0, 0, 0, 0]:
                merged_bbox = merged_bbox.copy()
                min_x1, min_y1, max_x2, max_y2 = merged_bbox
                has_valid_bbox = True
            else:
                min_x1 = min_y1 = max_x2 = max_y2 = 0
                has_valid_bbox = False
                # bbox가 없으면 기본값 설정
                merged_bbox = [0, 0, 0, 0]
            
            first_page_idx = merged_page_idx
            merged_token_count = num_tokens_from_string(merged_text)
            
            # 다음 블록들 확인
            j = i + 1
            
            while j < len(content_list):
                next_block = content_list[j]
                next_block_type = next_block.get("type", "").lower()
                next_text_level = next_block.get("text_level", 0)
                next_text = next_block.get("text", "")
                next_page_idx = next_block.get("page_idx", 0)

                # discarded 블록은 건너뛰고 계속 진행
                if next_block_type == "discarded":
                    j += 1
                    continue

                # table 블록이면 병합 중단 & caption 할당 처리
                if next_block_type == "table":
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}는 table, 병합 중단")
                    
                    # 이미 caption이 설정되어 있는지 확인 (별표 처리 등에서 설정됨)
                    existing_table_caption = next_block.get("table_caption", [])
                    existing_caption = next_block.get("caption", "")
                    
                    # table_caption이나 caption 중 하나라도 있으면 기존 caption으로 간주
                    has_existing_caption = bool(existing_table_caption) or bool(existing_caption)
                    caption_preview = str(existing_table_caption)[:50] if existing_table_caption else existing_caption[:50]
                    
                    if has_existing_caption:
                        self.logger.info(f"[MinerU-MERGE] 테이블 블록 {j}는 이미 caption이 있음 (별표 처리 등), 덮어쓰지 않음: {caption_preview}")
                    else:
                        # caption이 없을 때만 역방향 탐색
                        # table_caption이 리스트가 아니면 초기화
                        if not isinstance(next_block.get("table_caption"), list):
                            next_block["table_caption"] = []
                        
                        candidate_idx = None
                        candidate_text = None
                        
                        # 1. 같은 페이지에서 역방향 탐색 (discarded 건너뛰기)
                        for search_idx in range(j - 1, -1, -1):
                            search_block = content_list[search_idx]
                            search_type = search_block.get("type", "").lower()
                            search_page = search_block.get("page_idx", -1)
                            
                            # 페이지가 변경되면 중단
                            if search_page != next_page_idx:
                                break
                            
                            # discarded는 건너뛰기
                            if search_type == "discarded":
                                self.logger.info(f"[MinerU-MERGE]   search_idx={search_idx}: discarded, 건너뜀")
                                continue
                            
                            # text 블록 발견
                            if search_type == "text":
                                candidate_idx = search_idx
                                candidate_text = search_block.get("text", "").strip()
                                candidate_text_level = search_block.get("text_level", 0)
                                candidate_len = len(candidate_text)
                                
                                # text_level=0이고 길이 > 80이면 건너뛰기
                                if candidate_text_level == 0 and candidate_len > 80:
                                    self.logger.info(f"[MinerU-MERGE]   search_idx={search_idx}: text_level=0, len={candidate_len} > 80, 건너뜀")
                                    continue
                                
                                # 유효한 caption 후보 발견
                                self.logger.info(f"[MinerU-MERGE]   search_idx={search_idx}: text 발견 (같은 페이지), level={candidate_text_level}, len={candidate_len}")
                                break
                        
                        # 2. 같은 페이지에 없으면 바로 이전 페이지의 마지막 text 블록 검사
                        if not candidate_text:
                            self.logger.info(f"[MinerU-MERGE]   같은 페이지에 caption 후보 없음, 이전 페이지 검색")
                            
                            # 이전 페이지의 마지막 text 블록 찾기
                            prev_page_idx = next_page_idx - 1
                            
                            for search_idx in range(j - 1, -1, -1):
                                search_block = content_list[search_idx]
                                search_type = search_block.get("type", "").lower()
                                search_page = search_block.get("page_idx", -1)
                                
                                # 이전 페이지보다 더 앞으로 가면 중단
                                if search_page < prev_page_idx:
                                    break
                                
                                # 이전 페이지가 아니면 건너뛰기
                                if search_page != prev_page_idx:
                                    continue
                                
                                # text 블록 발견
                                if search_type == "text":
                                    candidate_idx = search_idx
                                    candidate_text = search_block.get("text", "").strip()
                                    candidate_text_level = search_block.get("text_level", 0)
                                    candidate_len = len(candidate_text)
                                    candidate_bbox = search_block.get("bbox", [0, 0, 0, 0])
                                    
                                    # text_level=0이고 길이 > 80이면 건너뛰기
                                    if candidate_text_level == 0 and candidate_len > 80:
                                        self.logger.info(f"[MinerU-MERGE]   search_idx={search_idx}: 이전 페이지, text_level=0, len={candidate_len} > 80, 건너뜀")
                                        continue
                                    
                                    # 유효한 caption 후보 발견 (이전 페이지의 마지막 text 블록)
                                    self.logger.info(f"[MinerU-MERGE]   search_idx={search_idx}: text 발견 (이전 페이지 마지막), level={candidate_text_level}, len={candidate_len}, bbox={candidate_bbox}")
                                    break
                        
                        # Caption 할당
                        if candidate_text:
                            next_block["table_caption"].append(candidate_text)
                            content_list[candidate_idx]["_used_as_caption"] = True
                            self.logger.info(f"[MinerU-MERGE] 블록 {candidate_idx}(level={content_list[candidate_idx].get('text_level', 0)}, len={len(candidate_text)})을 table 블록 {j}의 caption으로 할당: {candidate_text[:50]}")
                        else:
                            self.logger.info(f"[MinerU-MERGE] Caption 후보를 찾지 못함 (같은 페이지 및 이전 페이지 모두)")
                    
                    break

                # text 블록이 아니면 중단
                if next_block_type != "text":
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}는 {next_block_type}, 병합 중단")
                    break

                # 패턴이 있으면 중단
                if has_pattern(next_text):
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}에서 패턴 발견, 병합 중단")
                    break

                # 타이틀 블록 연속 병합 허용: 현재 블록이 title이고 다음 블록도 title이면 계속 진행
                if next_text_level > 0:
                    self.logger.info(f"[MinerU-MERGE] 블록 {j}는 title (level {next_text_level}), 병합 중단")
                    break

                # 토큰 수 체크
                next_token_count = num_tokens_from_string(next_text)
                if merged_token_count + next_token_count > max_tokens:
                    self.logger.info(f"[MinerU-MERGE] 토큰 수 초과 ({merged_token_count + next_token_count} > {max_tokens}), 병합 중단")
                    break

                # 병합 수행
                merged_text += "\n" + next_text
                merged_text_list.append(next_text)
                merged_token_count += next_token_count

                # bbox 업데이트
                next_bbox = next_block.get("bbox")
                if has_valid_bbox and next_bbox and len(next_bbox) == 4:
                    # 페이지가 변경되면 첫 블록의 bbox로 유지
                    if next_page_idx == first_page_idx:
                        # 같은 페이지면 bbox 확장
                        x1, y1, x2, y2 = next_bbox
                        min_x1 = min(min_x1, x1)
                        min_y1 = min(min_y1, y1)
                        max_x2 = max(max_x2, x2)
                        max_y2 = max(max_y2, y2)

                j += 1
            
            # 병합된 블록 생성
            if has_valid_bbox:
                merged_bbox = [min_x1, min_y1, max_x2, max_y2]
            else:
                # bbox가 유효하지 않으면 기본값 보장
                merged_bbox = [0, 0, 0, 0]
            
            merged_block = {
                "type": "text",
                "text": merged_text,
                "text_level": 0,
                "bbox": merged_bbox,
                "page_idx": first_page_idx
            }
            
            # 원본 블록의 다른 속성 복사
            for key in block:
                if key not in ["text", "type", "text_level", "bbox", "page_idx"]:
                    merged_block[key] = block[key]
            
            # 병합이 실제로 일어났는지 확인
            if j > i + 1:
                # 여러 블록이 병합됨 - merged_block 사용
                final_merged.append(merged_block)
                self.logger.info(f"[MinerU-MERGE] 블록 {i}~{j-1} 병합 완료 ({j-i}개 블록, {merged_token_count} 토큰)")
            else:
                # 병합 안 됨 - 원본 블록 유지
                final_merged.append(block)
            
            i = j
        
        # caption으로 사용된 블록 제외
        original_count = len(final_merged)
        final_merged = [b for b in final_merged if not b.get("_used_as_caption", False)]
        filtered_count = original_count - len(final_merged)
        if filtered_count > 0:
            self.logger.info(f"[MinerU-MERGE] caption으로 사용된 블록 {filtered_count}개 제외")
        
        self.logger.info(f"[MinerU-MERGE] 2단계 완료: {len(content_list)} -> {len(final_merged)} blocks")
        return final_merged

    def _read_output(self, output_dir: Path, file_stem: str, method: str = "auto", backend: str = "pipeline") -> list[dict[str, Any]]:
        json_file = None
        subdir = None
        attempted = []

        # mirror MinerU's sanitize_filename to align ZIP naming
        def _sanitize_filename(name: str) -> str:
            sanitized = re.sub(r"[/\\\.]{2,}|[/\\]", "", name)
            sanitized = re.sub(r"[^\w.-]", "_", sanitized, flags=re.UNICODE)
            if sanitized.startswith("."):
                sanitized = "_" + sanitized[1:]
            return sanitized or "unnamed"

        safe_stem = _sanitize_filename(file_stem)
        allowed_names = {f"{file_stem}_content_list.json", f"{safe_stem}_content_list.json"}
        self.logger.info(f"[MinerU] Expected output files: {', '.join(sorted(allowed_names))}")
        self.logger.info(f"[MinerU] Searching output in: {output_dir}")

        jf = output_dir / f"{file_stem}_content_list.json"
        self.logger.info(f"[MinerU] Trying original path: {jf}")
        attempted.append(jf)
        if jf.exists():
            subdir = output_dir
            json_file = jf
        else:
            alt = output_dir / f"{safe_stem}_content_list.json"
            self.logger.info(f"[MinerU] Trying sanitized filename: {alt}")
            attempted.append(alt)
            if alt.exists():
                subdir = output_dir
                json_file = alt
            else:
                nested_alt = output_dir / safe_stem / f"{safe_stem}_content_list.json"
                self.logger.info(f"[MinerU] Trying sanitized nested path: {nested_alt}")
                attempted.append(nested_alt)
                if nested_alt.exists():
                    subdir = nested_alt.parent
                    json_file = nested_alt

        if not json_file:
            raise FileNotFoundError(f"[MinerU] Missing output file, tried: {', '.join(str(p) for p in attempted)}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            for key in ("img_path", "table_img_path", "equation_img_path"):
                if key in item and item[key]:
                    item[key] = str((subdir / item[key]).resolve())
        return data

    def _transfer_to_sections(self, outputs: list[dict[str, Any]]):
        sections = []
        tables = []
        images = []
        
        for output in outputs:
            output_type = output.get("type", "")
            
            if output_type == MinerUContentType.TEXT:
                text = output.get("text", "")
                if text:
                    sections.append((text, self._line_tag(output)))
            
            elif output_type == MinerUContentType.TABLE:
                # caption/body 사이에 '\n\n' 추가, body는 markdownify로 변환
                from markdownify import markdownify as md
                
                # 1. 별표 처리 등에서 설정한 caption 필드 우선 사용
                existing_caption = output.get("caption", "").strip()
                
                # 2. table_caption 리스트 사용 (merge 단계에서 추가됨)
                table_caption_list = output.get("table_caption", [])
                table_caption_from_list = "\n".join(table_caption_list) if table_caption_list else ""
                
                # 3. 최종 caption 결정: existing_caption이 있으면 우선 사용, 없으면 table_caption_list 사용
                if existing_caption:
                    final_caption = existing_caption
                else:
                    final_caption = table_caption_from_list
                
                table_body_html = output.get("table_body", "")
                table_body_md = md(table_body_html) if table_body_html else ""
                table_footnote = "\n".join(output.get("table_footnote", []))
                
                table_text = final_caption + "\n\n" + table_body_md + table_footnote
                if not table_text.strip():
                    table_text = "FAILED TO PARSE TABLE"
                
                sections.append((table_text, self._line_tag(output)))
                
                # tables 리스트에도 추가
                table_img_path = output.get("table_img_path", "")
                if table_img_path:
                    tables.append((final_caption, table_img_path, table_body_md, self._line_tag(output)))
            
            elif output_type == MinerUContentType.IMAGE:
                image_caption_list = output.get("image_caption", [])
                image_footnote_list = output.get("image_footnote", [])
                caption_text = "".join(image_caption_list)
                footnote_text = "".join(image_footnote_list)
                
                image_text = caption_text
                if footnote_text:
                    image_text = image_text + "\n" + footnote_text
                
                if image_text:
                    sections.append((image_text, self._line_tag(output)))
                
                # images 리스트에도 추가
                img_path = output.get("img_path", "")
                if img_path:
                    images.append((caption_text, img_path, self._line_tag(output)))
            
            elif output_type == MinerUContentType.EQUATION:
                text = output.get("text", "")
                if text:
                    sections.append((text, self._line_tag(output)))
            
            elif output_type == MinerUContentType.CODE:
                code_body = output.get("code_body", "")
                code_caption_list = output.get("code_caption", [])
                caption_text = "\n".join(code_caption_list)
                
                code_text = code_body
                if caption_text:
                    code_text = caption_text + "\n" + code_text
                
                if code_text:
                    sections.append((code_text, self._line_tag(output)))
            
            elif output_type == MinerUContentType.LIST:
                list_items = output.get("list_items", [])
                list_text = "\n".join(list_items)
                if list_text:
                    sections.append((list_text, self._line_tag(output)))
            
            elif output_type == MinerUContentType.DISCARDED:
                continue  # Skip discarded blocks entirely
        
        return sections, tables, images

    def _transfer_to_tables(self, outputs: list[dict[str, Any]]):
        return []

    def _transfer_to_images(self, outputs: list[dict[str, Any]]) -> list:
        """
        MinerU 출력에서 IMAGE 블록을 추출하여 테이블과 동일한 형식으로 반환
        
        Returns:
            images: List[Tuple[Tuple[Image, str], List[Tuple]]]
                - (PIL.Image, caption_text): 이미지와 캡션
                - [(page_num, left, right, top, bottom), ...]: bbox 위치
        """
        image_data_list = []  # 임시로 이미지 데이터 저장 (img, base_caption, poss, page_idx)
        last_image_base_caption = None  # 이전 이미지의 기본 캡션 (bbox 제외)
        last_image_page = None  # 이전 이미지의 페이지 번호
        
        for idx, output in enumerate(outputs):
            output_type = output.get("type", "").lower()
            
            if output_type == "image":
                # 1. 이미지 파일 로드
                img_path = output.get("img_path")
                if not img_path or not os.path.exists(img_path):
                    self.logger.warning(f"[MinerU-IMAGE] Image file not found: {img_path}")
                    continue
                
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    self.logger.error(f"[MinerU-IMAGE] Failed to load image {img_path}: {e}")
                    continue
                
                # 2. 캡션 텍스트 추출
                caption = "".join(output.get("image_caption", []))
                footnote = "".join(output.get("image_footnote", []))
                text = caption + ("\n" + footnote if footnote else "")
                
                # 3. bbox 위치 추출 및 실제 픽셀 좌표로 변환
                bbox = output.get("bbox")
                page_idx = output.get("page_idx", 0)
                
                if bbox and len(bbox) == 4:
                    # bbox는 정규화된 좌표 (0-1000) - 실제 픽셀 좌표로 변환 필요
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox
                    
                    # PDF 페이지 크기 기준으로 정규화 해제
                    if hasattr(self, "pdf") and self.pdf and self.pdf.pages and page_idx < len(self.pdf.pages):
                        pdf_page = self.pdf.pages[page_idx]
                        page_width = pdf_page.width
                        page_height = pdf_page.height
                        
                        # 정규화된 좌표 (0-1000)를 실제 픽셀 좌표로 변환
                        x1 = (x1_norm / 1000.0) * page_width
                        y1 = (y1_norm / 1000.0) * page_height
                        x2 = (x2_norm / 1000.0) * page_width
                        y2 = (y2_norm / 1000.0) * page_height
                        
                        poss = [(page_idx, x1, x2, y1, y2)]
                        self.logger.debug(f"[MinerU-IMAGE] page={page_idx}, "
                                        f"bbox_norm=({x1_norm:.1f},{y1_norm:.1f},{x2_norm:.1f},{y2_norm:.1f}), "
                                        f"bbox_pixel=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
                                        f"pdf_size=({page_width:.1f}x{page_height:.1f})")
                    else:
                        # PDF 정보가 없으면 정규화된 좌표 그대로 사용 (폴백)
                        x1, y1, x2, y2 = x1_norm, y1_norm, x2_norm, y2_norm
                        poss = [(page_idx, x1, x2, y1, y2)]
                        self.logger.warning(f"[MinerU-IMAGE] PDF info not available for page {page_idx}, using normalized coords")
                else:
                    # bbox 없으면 빈 리스트
                    poss = []
                    self.logger.warning(f"[MinerU-IMAGE] No bbox for image on page {page_idx}")
                
                # 4. 캡션이 없으면 bbox 거리 기반으로 가장 가까운 텍스트 블록 찾기
                base_caption = None  # 인덱스 제외한 기본 캡션
                match_type = None  # 매칭 유형 (위쪽/좌측/우측)
                
                if not text.strip():
                    # 4-1. 직전 블록이 같은 페이지의 이미지 블록이면 그 캡션 계승
                    if idx > 0:
                        prev_block = outputs[idx - 1]
                        prev_type = prev_block.get("type", "").lower()
                        prev_page = prev_block.get("page_idx", -1)
                        
                        if prev_type == "image" and prev_page == page_idx and last_image_base_caption:
                            # 같은 페이지의 연속된 이미지 블록 - 이전 이미지의 기본 캡션 계승
                            base_caption = last_image_base_caption
                            match_type = "inherit"
                            self.logger.info(f"[MinerU-IMAGE] Inheriting caption from previous image: page={page_idx}, caption={base_caption[:60]}")
                    
                    # 4-2. bbox 거리 기반으로 같은 페이지에서 가장 가까운 텍스트 블록 찾기
                    if not base_caption and bbox and len(bbox) == 4:
                        # 이미지의 bbox 좌표 (정규화된 좌표)
                        img_x1, img_y1, img_x2, img_y2 = bbox
                        
                        nearest_text = None
                        min_distance = float('inf')
                        nearest_match_type = None
                        best_priority = 999  # 현재 최고 우선순위 (낮을수록 우선)
                        
                        # 같은 페이지의 모든 텍스트 블록과 거리 비교
                        for other_idx, other_block in enumerate(outputs):
                            other_type = other_block.get("type", "").lower()
                            other_page = other_block.get("page_idx", -1)
                            
                            # 같은 페이지의 텍스트 블록만 검사
                            if other_type != "text" or other_page != page_idx:
                                continue
                            
                            other_bbox = other_block.get("bbox")
                            if not other_bbox or len(other_bbox) != 4:
                                continue
                            
                            # 텍스트 블록의 bbox 좌표
                            text_x1, text_y1, text_x2, text_y2 = other_bbox
                            
                            distance = None
                            current_match_type = None
                            priority = 999  # 우선순위 (낮을수록 우선)
                            
                            # 1. 좌측 관계: 이미지가 텍스트 우측에 있는 경우 (img_x1 > text_x2) - 최우선
                            if img_x1 > text_x2:
                                # Y축 겹침이 있는지 확인 (같은 라인에 있는지)
                                y_overlap = not (img_y2 < text_y1 or img_y1 > text_y2)
                                if y_overlap:
                                    distance = img_x1 - text_x2
                                    current_match_type = "left"
                                    priority = 1  # 최고 우선순위
                            
                            # 2. 우측 관계: 이미지가 텍스트 좌측에 있는 경우 (img_x2 < text_x1) - 차선
                            elif img_x2 < text_x1:
                                # Y축 겹침이 있는지 확인 (같은 라인에 있는지)
                                y_overlap = not (img_y2 < text_y1 or img_y1 > text_y2)
                                if y_overlap:
                                    distance = text_x1 - img_x2
                                    current_match_type = "right"
                                    priority = 2  # 두 번째 우선순위
                            
                            # 3. 위쪽 관계: 텍스트가 이미지 위쪽에 있는 경우 (text_y2 < img_y1) - 최후
                            elif text_y2 < img_y1:
                                distance = img_y1 - text_y2
                                current_match_type = "above"
                                priority = 3  # 가장 낮은 우선순위
                            
                            # 가장 가까운 거리 업데이트 (우선순위 우선, 같은 우선순위면 거리로 판단)
                            if distance is not None:
                                # 우선순위가 더 높거나(낮은 숫자), 같은 우선순위에서 거리가 더 가까우면 업데이트
                                if priority < best_priority or (priority == best_priority and distance < min_distance):
                                    min_distance = distance
                                    best_priority = priority
                                    nearest_text = other_block.get("text", "").strip()
                                    nearest_match_type = current_match_type
                        
                        # 가장 가까운 텍스트 블록이 있으면 사용
                        if nearest_text:
                            base_caption = nearest_text
                            match_type = nearest_match_type
                            self.logger.info(f"[MinerU-IMAGE] Found nearest text by bbox: page={page_idx}, "
                                          f"match_type={match_type}, distance={min_distance:.1f}, "
                                          f"caption={base_caption[:60]}")
                    
                    # 4-3. base_caption 설정 (suffix 없이)
                    if not base_caption:
                        # 가까운 텍스트를 찾지 못하면 폴백
                        base_caption = f"페이지 {page_idx + 1}"
                        match_type = "fallback"
                        self.logger.info(f"[MinerU-IMAGE] No nearby text block found, using fallback caption: {base_caption}")
                else:
                    # MinerU가 추출한 캡션이 있으면 그대로 사용
                    base_caption = text
                
                # 5. 현재 이미지의 기본 캡션을 다음 이미지를 위해 저장
                last_image_base_caption = base_caption
                last_image_page = page_idx
                
                # 6. 이미지 데이터를 임시 리스트에 저장 (최종 캡션은 나중에 추가)
                image_data_list.append({
                    "img": img,
                    "base_caption": base_caption,
                    "poss": poss,
                    "page_idx": page_idx,
                    "caption_len": len(caption)
                })
                self.logger.info(f"[MinerU-IMAGE] Extracted image: page={page_idx}, base_caption={base_caption[:60]}, size={img.size}")
        
        # 7. 최종 images 리스트 생성: base_caption + [이미지 {리스트 인덱스}]
        images = []
        for list_idx, img_data in enumerate(image_data_list):
            img = img_data["img"]
            base_caption = img_data["base_caption"]
            poss = img_data["poss"]
            
            # 리스트 인덱스를 사용하여 최종 캡션 생성
            final_caption = f"{base_caption} [이미지 {list_idx}]"
            
            images.append(((img, final_caption), poss))
            self.logger.info(f"[MinerU-IMAGE] Final caption [{list_idx}]: {final_caption[:80]}")
        
        self.logger.info(f"[MinerU-IMAGE] Total images extracted: {len(images)}")
        return images

    def parse_pdf(
            self,
            filepath: str | PathLike[str],
            binary: BytesIO | bytes,
            callback: Optional[Callable] = None,
            *,
            output_dir: Optional[str] = None,
            backend: str = "pipeline",
            server_url: Optional[str] = None,
            delete_output: bool = True,
            parse_method: str = "raw",
            **kwargs,
    ) -> tuple:
        import shutil

        temp_pdf = None
        created_tmp_dir = False

        parser_cfg = kwargs.get('parser_config', {})
        
        # Extract mineru-specific settings from nested 'mineru' key
        mineru_settings = parser_cfg.get('mineru', {})
        
        # Language: try mineru settings, then parser_cfg, then kwargs, finally default
        lang = mineru_settings.get('lang') or parser_cfg.get('mineru_lang') or kwargs.get('lang', 'English')
        mineru_lang_code = LANGUAGE_TO_MINERU_MAP.get(lang, 'ch')  # Defaults to Chinese if not matched
        
        # Parse method: check both 'method' and 'parse_method' in mineru settings
        mineru_method_raw_str = mineru_settings.get('method') or mineru_settings.get('parse_method') or parser_cfg.get('mineru_parse_method', 'auto')
        
        # Formula and table: use mineru settings if available, with explicit defaults
        enable_formula = mineru_settings.get('formula', mineru_settings.get('formula_enable', parser_cfg.get('mineru_formula_enable', True)))
        enable_table = mineru_settings.get('table', mineru_settings.get('table_enable', parser_cfg.get('mineru_table_enable', True)))
        
        # Max tokens for text merging: use mineru settings if available, default to 512
        max_tokens = mineru_settings.get('max_tokens', parser_cfg.get('mineru_max_tokens', 512))
        
        self.logger.info(f"[MinerU] Extracted options - lang={lang}, lang_code={mineru_lang_code}, method={mineru_method_raw_str}, formula={enable_formula}, table={enable_table}, max_tokens={max_tokens}")

        # remove spaces, or mineru crash, and _read_output fail too
        file_path = Path(filepath)
        # XXX: Preserve original extension instead of forcing .pdf
        file_ext = file_path.suffix or ".pdf"  # Default to .pdf if no extension
        pdf_file_name = file_path.stem.replace(" ", "") + file_ext
        pdf_file_path_valid = os.path.join(file_path.parent, pdf_file_name)

        if binary:
            temp_dir = Path(tempfile.mkdtemp(prefix="mineru_bin_pdf_"))
            temp_pdf = temp_dir / pdf_file_name
            with open(temp_pdf, "wb") as f:
                f.write(binary)
            pdf = temp_pdf
            self.logger.info(f"[MinerU] Received binary PDF -> {temp_pdf}")
            if callback:
                callback(0.15, f"[MinerU] Received binary PDF -> {temp_pdf}")
        else:
            if pdf_file_path_valid != filepath:
                self.logger.info(f"[MinerU] Remove all space in file name: {pdf_file_path_valid}")
                shutil.move(filepath, pdf_file_path_valid)
            pdf = Path(pdf_file_path_valid)
            if not pdf.exists():
                if callback:
                    callback(-1, f"[MinerU] PDF not found: {pdf}")
                raise FileNotFoundError(f"[MinerU] PDF not found: {pdf}")

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_pdf_"))
            created_tmp_dir = True

        self.logger.info(f"[MinerU] Output directory: {out_dir} backend={backend} api={self.mineru_api} server_url={server_url or self.mineru_server_url}")
        if callback:
            callback(0.15, f"[MinerU] Output directory: {out_dir}")

        # Only extract PDF page images if the file is actually a PDF
        # Skip for image files (PNG, JPG, etc.) to avoid pdfplumber errors
        file_ext = pdf.suffix.lower()
        if file_ext == '.pdf':
            self.__images__(pdf, zoomin=1)
        else:
            self.logger.info(f"[MinerU] Skipping __images__ for non-PDF file: {file_ext}")
            self.page_images = None
            self.pdf = None

        try:
            options = MinerUParseOptions(
                backend=MinerUBackend(backend),
                lang=MinerULanguage(mineru_lang_code),
                method=MinerUParseMethod(mineru_method_raw_str),
                server_url=server_url,
                delete_output=delete_output,
                parse_method=parse_method,
                formula_enable=enable_formula,
                table_enable=enable_table,
            )
            final_out_dir = self._run_mineru(pdf, out_dir, options, callback=callback)
            outputs = self._read_output(final_out_dir, pdf.stem, method=mineru_method_raw_str, backend=backend)
            self.logger.info(f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            self._show_mineru_raw_blocks(outputs, max_blocks=100, page_start=1, page_end=20)

            # 블록 전처리: header/footer 후보 감지 및 discarded 표시
            header_candidates, footer_candidates = self._preprocess_blocks(outputs)
            
            # 커스텀 병합: text 블록 병합 (max_tokens 전달)
            merged_outputs = self._custom_merge(outputs, max_tokens=max_tokens)
            
            # sections, tables, images 추출           
            sections, tables, images = self._transfer_to_sections(merged_outputs)
            tables = self._transfer_to_tables(merged_outputs)
            images = self._transfer_to_images(outputs)  # 병합 전 outputs 사용
            
            if callback:
                callback(0.75, f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            return sections, tables, images
        finally:
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = MinerUParser("mineru")
    ok, reason = parser.check_installation()
    print("MinerU available:", ok)

    filepath = ""
    with open(filepath, "rb") as file:
        outputs = parser.parse_pdf(filepath=filepath, binary=file.read())
        for output in outputs:
            print(output)
