# Dataset Metadata 설정과 Chat Metadata 필터링의 관계

## 개요

RAGFlow의 **Metadata** 기능은 두 단계로 작동합니다:

1. **Dataset Configuration**: 문서 파싱 시 자동으로 메타데이터 추출 (Auto-extract metadata)
2. **Chat Configuration**: 검색 시 메타데이터 기반 필터링 (Metadata filtering)

이 두 단계는 서로 **독립적이면서도 연결**되어 있습니다.

---

## 1단계: Dataset Configuration - 메타데이터 추출

### 1.1 설정 위치

**프론트엔드**: [web/src/pages/dataset/dataset-setting/configuration/common-item.tsx](web/src/pages/dataset/dataset-setting/configuration/common-item.tsx#L376-L450)

```tsx
export function AutoMetadata({
  type = MetadataType.Setting,
  otherData,
}: {
  type?: MetadataType;
  otherData?: Record<string, any>;
}) {
  // 메타데이터 설정 다이얼로그 열기
  const handleClickOpenMetadata = useCallback(() => {
    const metadata = form.getValues('parser_config.metadata');
    const builtInMetadata = form.getValues('parser_config.built_in_metadata');
    const tableMetaData = util.metaDataSettingJSONToMetaDataTableData(metadata);
    showManageMetadataModal({
      metadata: tableMetaData,
      isCanAdd: true,
      type: type,
      record: otherData,
      builtInMetadata,
    });
  }, [form, otherData, showManageMetadataModal, type]);
  
  // 메타데이터 저장
  const handleSaveMetadata = (data?: {
    metadata?: IMetaDataReturnJSONSettings;
    builtInMetadata?: IBuiltInMetadataItem[];
  }) => {
    form.setValue('parser_config.metadata', data?.metadata || []);
    form.setValue('parser_config.built_in_metadata', data?.builtInMetadata || []);
    form.setValue('parser_config.enable_metadata', true);
  };
}
```

### 1.2 메타데이터 스키마 구조

**타입 정의**: [web/src/pages/dataset/components/metedata/interface.ts](web/src/pages/dataset/components/metedata/interface.ts#L1-L43)

```typescript
// JSON Schema 형식으로 저장됨
export interface IMetaDataJsonSchema {
  type?: 'object';
  properties?: Record<string, IMetaDataJsonSchemaProperty>;
  additionalProperties?: boolean;
}

export interface IMetaDataJsonSchemaProperty {
  type?: string;  // 'string', 'int', 'float', 'bool', 'time'
  description?: string;  // LLM에게 제공할 설명
  enum?: string[];  // 제한된 값 목록 (Enum Mode)
  items?: {
    type?: string;
    enum?: string[];
  };
  format?: string;
}
```

### 1.2.1 메타데이터 스키마 설계 시 주의사항

**중요**: 메타데이터는 **청크 단위**로 LLM이 추출하지만, **문서 단위**로 저장됩니다.

따라서 스키마 설계 시:
- ✅ **청크에서 식별 가능한 정보**: author, topic, keywords, date 등
- ⚠️ **문서 전체 분석이 필요한 정보**: category, document_type, summary 등
  - 이런 필드는 여러 청크에서 일관된 값이 추출되도록 설계해야 함
  - description에 명확한 지침 제공 필요

**좋은 예시** (청크에서 추출 가능):
```json
{
  "type": "object",
  "properties": {
    "author": {
      "type": "string",
      "description": "The author mentioned in this text. Look for names after 'by', 'written by', 'Author:', etc.",
      "enum": ["Alice", "Bob", "Charlie"]
    },
    "topic": {
      "type": "string",
      "description": "Main topic discussed in this section (AI, ML, RAG, NLP, etc.)"
    },
    "date_mentioned": {
      "type": "string",
      "format": "time",
      "description": "Any date explicitly mentioned in this text (YYYY-MM-DD format)"
    }
  }
}
```

**주의가 필요한 예시** (문서 전체 판단 필요):
```json
{
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "description": "Document type: 'research' if discussing research/experiments, 'manual' if providing instructions, 'report' if summarizing findings. Based on the overall tone and structure of the text.",
      "enum": ["research", "manual", "report", "guide"]
    }
  }
}
```
⚠️ 위 예시의 "category"는 청크마다 다르게 해석될 수 있어 부정확할 수 있습니다.

### 1.3 파싱 시 메타데이터 자동 추출

#### 1.3.1 청크별 추출 및 병합 메커니즘

**핵심 원리**: 
1. LLM이 **각 청크의 내용만** 보고 메타데이터 추출
2. 모든 청크의 결과를 `update_metadata_to()` 함수로 **병합**
3. 병합 시 **리스트 타입은 누적**, **문자열 타입은 덮어쓰기**

**백엔드**: [rag/svr/task_executor.py](rag/svr/task_executor.py#L401-L447)

```python
# 문서 파싱 후 청크 생성 시 실행
if task["parser_config"].get("enable_metadata", False) and task["parser_config"].get("metadata"):
    st = timer()
    progress_callback(msg="Start to generate meta-data for every chunk ...")
    
    # Chat 모델 준비 (Indexing model)
    chat_mdl = LLMBundle(task["tenant_id"], LLMType.CHAT, 
                        llm_name=task["llm_id"], lang=task["language"])

    async def gen_metadata_task(chat_mdl, d):
        # 캐시 확인
        cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], 
                              "metadata", task["parser_config"]["metadata"])
        if not cached:
            if has_canceled(task["id"]):
                progress_callback(-1, msg="Task has been canceled.")
                return
            async with chat_limiter:
                # LLM으로 메타데이터 추출
                cached = await gen_metadata(
                    chat_mdl,
                    metadata_schema(task["parser_config"]["metadata"]),
                    d["content_with_weight"]
                )
            set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, 
                         "metadata", task["parser_config"]["metadata"])
        if cached:
            d["metadata_obj"] = cached

    # 모든 청크에 대해 병렬 처리
    tasks = []
    for d in docs:
        tasks.append(asyncio.create_task(gen_metadata_task(chat_mdl, d)))
    
    await asyncio.gather(*tasks, return_exceptions=False)
    
    # 메타데이터 병합 및 저장
    metadata = {}
    for doc in docs:
        # 각 청크의 메타데이터를 병합 (핵심!)
        metadata = update_metadata_to(metadata, doc["metadata_obj"])
        del doc["metadata_obj"]
    
    if metadata:
        e, doc = DocumentService.get_by_id(task["doc_id"])
        if e:
            if isinstance(doc.meta_fields, str):
                doc.meta_fields = json.loads(doc.meta_fields)
            # 기존 문서의 메타데이터와도 병합
            metadata = update_metadata_to(metadata, doc.meta_fields)
            # Document의 meta_fields에 저장
            DocumentService.update_by_id(task["doc_id"], {"meta_fields": metadata})
```

#### 1.3.2 병합 로직 상세 분석

**코드**: [common/metadata_utils.py](common/metadata_utils.py#L180-L211)

```python
def update_metadata_to(metadata, meta):
    """
    청크별 메타데이터를 문서 메타데이터로 병합
    
    병합 규칙:
    1. 리스트 타입: 모든 값을 누적 (중복 제거)
    2. 문자열 타입: 마지막 값으로 덮어쓰기
    3. 리스트 + 문자열 혼합: 리스트에 추가
    """
    if not meta:
        return metadata
    
    # JSON 문자열이면 파싱
    if isinstance(meta, str):
        try:
            meta = json_repair.loads(meta)
        except Exception:
            logging.error("Meta data format error.")
            return metadata
    
    if not isinstance(meta, dict):
        return metadata

    for k, v in meta.items():
        # 리스트 정리: 문자열만 유지, 중복 제거
        if isinstance(v, list):
            v = [vv for vv in v if isinstance(vv, str)]
            if not v:
                continue
            v = dedupe_list(v)  # 중복 제거
        
        # 리스트와 문자열만 허용
        if not isinstance(v, list) and not isinstance(v, str):
            continue
        
        # 첫 번째 청크: 그대로 저장
        if k not in metadata:
            metadata[k] = v
            continue
        
        # 이후 청크: 병합
        if isinstance(metadata[k], list):
            # 기존이 리스트면 추가
            if isinstance(v, list):
                metadata[k].extend(v)  # 리스트 확장
            else:
                metadata[k].append(v)  # 문자열 추가
            metadata[k] = dedupe_list(metadata[k])  # 중복 제거
        else:
            # 기존이 문자열이면 덮어쓰기
            metadata[k] = v

    return metadata
```

#### 1.3.3 청크별 추출 시뮬레이션

**시나리오**: 3개 청크로 구성된 연구 논문

**스키마**:
```json
{
  "properties": {
    "author": {"type": "string"},
    "topic": {"type": "string"},
    "keywords": {"type": "array", "items": {"type": "string"}}
  }
}
```

**청크 1 (도입부)**:
```text
"This paper by Alice explores Retrieval-Augmented Generation (RAG).
Keywords: information retrieval, language models."
```
→ LLM 추출: `{"author": "Alice", "topic": "RAG", "keywords": ["information retrieval", "language models"]}`

**청크 2 (본문)**:
```text
"The methodology section discusses embedding techniques and vector search.
Keywords: embeddings, similarity search."
```
→ LLM 추출: `{"topic": "RAG", "keywords": ["embeddings", "similarity search"]}`

**청크 3 (결론)**:
```text
"Alice concludes that RAG improves accuracy in question answering tasks.
Keywords: question answering, accuracy."
```
→ LLM 추출: `{"author": "Alice", "topic": "RAG", "keywords": ["question answering", "accuracy"]}`

**병합 과정**:
```python
# 초기 상태
metadata = {}

# 청크 1 병합
metadata = update_metadata_to(metadata, {
    "author": "Alice", 
    "topic": "RAG", 
    "keywords": ["information retrieval", "language models"]
})
# 결과: {"author": "Alice", "topic": "RAG", "keywords": ["information retrieval", "language models"]}

# 청크 2 병합
metadata = update_metadata_to(metadata, {
    "topic": "RAG", 
    "keywords": ["embeddings", "similarity search"]
})
# 결과: {
#   "author": "Alice",  # 유지
#   "topic": "RAG",  # 덮어쓰기 (같은 값)
#   "keywords": ["information retrieval", "language models", "embeddings", "similarity search"]  # 누적
# }

# 청크 3 병합
metadata = update_metadata_to(metadata, {
    "author": "Alice", 
    "topic": "RAG", 
    "keywords": ["question answering", "accuracy"]
})
# 최종 결과: {
#   "author": "Alice",  # 덮어쓰기 (같은 값)
#   "topic": "RAG",  # 덮어쓰기 (같은 값)
#   "keywords": [
#     "information retrieval", 
#     "language models", 
#     "embeddings", 
#     "similarity search",
#     "question answering",
#     "accuracy"
#   ]  # 모든 청크의 키워드 누적
# }
```

#### 1.3.4 문제점과 해결 방안

**문제 1: 청크만 보고 문서 타입 판단의 어려움**

예시: "category" 필드로 문서 타입 구분
```json
{
  "category": {
    "type": "string",
    "enum": ["research", "manual", "report"]
  }
}
```

**문제 시나리오**:
- 청크 1 (도입부): "This research paper..." → `"category": "research"`
- 청크 2 (방법론): "Follow these steps..." → `"category": "manual"` (절차 설명으로 오인)
- 청크 3 (결론): "In summary, the report..." → `"category": "report"`
- **최종 결과**: `"category": "report"` (마지막 값으로 덮어쓰기됨) ❌

**해결 방안**:

**방법 1: 리스트 타입으로 변경하여 투표 방식**
```json
{
  "category": {
    "type": "array",
    "items": {
      "type": "string",
      "enum": ["research", "manual", "report"]
    }
  }
}
```
→ 결과: `["research", "manual", "report"]` → 가장 많이 나온 값 선택 (후처리 필요)

**방법 2: Description 개선으로 LLM 정확도 향상**
```json
{
  "category": {
    "type": "string",
    "description": "Determine ONLY if this entire text segment belongs to: 'research' (experimental/academic), 'manual' (step-by-step instructions), or 'report' (summary/findings). If unsure, do not extract.",
    "enum": ["research", "manual", "report"]
  }
}
```

**방법 3: Built-in Metadata 사용** (파일 메타데이터 기반)
- `parser_config.built_in_metadata`: 파일 이름, 확장자 등에서 추출
- 청크 내용이 아닌 문서 속성으로 판단

**방법 4: Dataset별 구분 메타데이터 추가**
```json
{
  "category": {
    "type": "string",
    "enum": ["research", "manual", "report"]
  },
  "source_dataset": {
    "type": "string",
    "description": "Name of the source dataset for filtering purposes",
    "enum": ["Medical_Research", "Technical_Manuals"]
  }
}
```
→ 파싱 후 SDK로 `source_dataset` 메타데이터 일괄 설정하여 Dataset 구분 가능

**문제 2: 일관성 없는 추출**

예시: "author" 필드
- 청크 1: `"author": "Alice Smith"`
- 청크 2: `"author": "A. Smith"`
- 청크 3: `"author": "Alice"`
- **최종 결과**: `"author": "Alice"` (마지막 값만 유지) ⚠️

**해결 방안: Enum Mode 사용**
```json
{
  "author": {
    "type": "string",
    "description": "Author name. Look for 'by', 'Author:', etc. Extract full name if available.",
    "enum": ["Alice Smith", "Bob Johnson", "Charlie Brown"]
  }
}
```
→ LLM은 enum 리스트의 값만 선택 가능 → 일관성 보장

**LLM 프롬프트에서의 Enum 처리**:
```python
# rag/prompts/generator.py
for k, desc in schema["properties"].items():
    if desc.get("enum"):
        desc["description"] += "\n** Extracted values must strictly match the given list specified by `enum`. **"
```

#### 1.3.5 LLM 프롬프트 분석

**프롬프트**: [rag/prompts/meta_data.md](rag/prompts/meta_data.md)

```markdown
## Role: Metadata extraction expert.
## Rules:
 - Strict Evidence Only: Extract a value ONLY if it is explicitly mentioned in the Content. 
 - Enum Filter: For any field with an 'enum' list, the list acts as a strict filter. 
   If no element from the list (or its direct synonym) is found in the Content, 
   you MUST NOT extract that field.
 - No Meta-Inference: Do not infer values based on the document's nature, format, or category. 
   If the text does not literally state the information, treat it as missing.
 - Zero-Hallucination: Never invent information or pick a "likely" value from the enum to fill a field.
 - Empty Result: If no matches are found for any field, or if the content is irrelevant, 
   output ONLY {}. 
 - Output: ONLY a valid JSON string. No Markdown, no notes.

## Schema for extraction:
{{ schema }}

## Content to analyze:
{{ content }}
```

**핵심 원칙**:
- ✅ **명시적 증거만**: 청크에 명확히 언급된 내용만 추출
- ✅ **추론 금지**: 문서 형식이나 맥락으로 추측하지 않음
- ✅ **Enum 엄격 준수**: enum 리스트 외 값은 추출하지 않음
- ✅ **환각 방지**: 정보가 없으면 빈 객체 `{}` 반환

**문서 타입 판단의 한계**:
청크 내용에 "This is a research paper"같은 명시적 언급이 없으면 LLM은 category를 추출하지 **않습니다**.
→ 따라서 문서 전체의 타입은 청크만으로 판단하기 어려움
```

**LLM 프롬프트**: [rag/prompts/generator.py](rag/prompts/generator.py#L934-L946)

```python
async def gen_metadata(chat_mdl, schema: dict, content: str):
    template = PROMPT_JINJA_ENV.from_string(META_DATA)
    
    # Enum 처리: 값이 없으면 제거
    for k, desc in schema["properties"].items():
        if "enum" in desc and not desc.get("enum"):
            del desc["enum"]
        # Enum이 있으면 설명에 강조 추가
        if desc.get("enum"):
            desc["description"] += "\n** Extracted values must strictly match the given list specified by `enum`. **"
    
    # 프롬프트 생성
    system_prompt = template.render(content=content, schema=schema)
    user_prompt = "Output: "
    
    # LLM 호출
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    
    # 응답 파싱
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
```

### 1.4 메타데이터 저장 구조

**DB 모델**: [api/db/db_models.py](api/db/db_models.py#L790)

```python
class Document(BaseModel):
    # ...
    meta_fields = JSONField(null=True, default={})  # JSON 필드로 저장
    # ...
```

**저장 예시**:
```json
{
  "author": "Alice",
  "publication_date": "2025-01-15",
  "category": "research",
  "tags": ["AI", "RAG", "LLM"]
}
```

---

## 2단계: Chat Configuration - 메타데이터 필터링

### 2.1 설정 위치

**프론트엔드**: [web/src/components/metadata-filter/index.tsx](web/src/components/metadata-filter/index.tsx#L1-L93)

```tsx
export function MetadataFilter({
  prefix = '',
  canReference,
}: MetadataFilterProps) {
  const form = useFormContext();

  const methodName = prefix + 'meta_data_filter.method';
  const kbIds: string[] = useWatch({ control: form.control, name: prefix + 'kb_ids' });
  const metadata = useWatch({ control: form.control, name: methodName });
  
  const hasKnowledge = Array.isArray(kbIds) && kbIds.length > 0;

  const MetadataOptions = Object.values(DatasetMetadata).map((x) => {
    return {
      value: x,  // 'disabled', 'auto', 'manual', 'semi_auto'
      label: t(`meta.${x}`),
    };
  });

  return (
    <>
      {hasKnowledge && (
        <RAGFlowFormItem label={t('metadata')} name={methodName} tooltip={t('metadataTip')}>
          <SelectWithSearch options={MetadataOptions} triggerClassName="!bg-bg-input" />
        </RAGFlowFormItem>
      )}
      {/* Manual Mode */}
      {hasKnowledge && metadata === DatasetMetadata.Manual && (
        <MetadataFilterConditions kbIds={kbIds} prefix={prefix} canReference={canReference} />
      )}
      {/* Semi-Automatic Mode */}
      {hasKnowledge && metadata === DatasetMetadata.SemiAutomatic && (
        <MetadataSemiAutoFields kbIds={kbIds} prefix={prefix} />
      )}
    </>
  );
}
```

### 2.2 세 가지 필터링 모드

#### Mode 1: Disabled (비활성화)
- 메타데이터 필터링을 사용하지 않음
- 모든 문서가 검색 대상

#### Mode 2: Automatic (자동)
**프론트엔드**: [web/src/components/metadata-filter/index.tsx](web/src/components/metadata-filter/index.tsx#L38-L65)

```tsx
// 사용자가 선택만 하면 LLM이 자동으로 필터 조건 생성
{hasKnowledge && (
  <RAGFlowFormItem label={t('metadata')} name={methodName}>
    <SelectWithSearch options={MetadataOptions} />
  </RAGFlowFormItem>
)}
```

**백엔드**: [common/metadata_utils.py](common/metadata_utils.py#L124-L146)

```python
async def apply_meta_data_filter(
    meta_data_filter: dict | None,
    metas: dict,  # KB의 모든 메타데이터
    question: str,
    chat_mdl: Any = None,
    base_doc_ids: list[str] | None = None,
    manual_value_resolver: Callable[[dict], dict] | None = None,
) -> list[str] | None:
    method = meta_data_filter.get("method")

    if method == "auto":
        # LLM이 사용자 질문과 메타데이터를 분석하여 필터 조건 자동 생성
        filters: dict = await gen_meta_filter(chat_mdl, metas, question)
        doc_ids.extend(meta_filter(metas, filters["conditions"], filters.get("logic", "and")))
        if not doc_ids:
            return None
```

**LLM 프롬프트**: [rag/prompts/generator.py](rag/prompts/generator.py#L516-L536)

```python
async def gen_meta_filter(chat_mdl, meta_data: dict, query: str) -> dict:
    # 메타데이터 구조 준비
    meta_data_structure = {}
    for key, values in meta_data.items():
        meta_data_structure[key] = list(values.keys()) if isinstance(values, dict) else values

    # 프롬프트 생성
    sys_prompt = PROMPT_JINJA_ENV.from_string(META_FILTER).render(
        current_date=datetime.datetime.today().strftime('%Y-%m-%d'),
        metadata_keys=json.dumps(meta_data_structure),
        user_question=query
    )
    user_prompt = "Generate filters:"
    
    # LLM 호출
    ans = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": user_prompt}])
    ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
    
    try:
        ans = json_repair.loads(ans)
        assert isinstance(ans, dict), ans
        assert "conditions" in ans and isinstance(ans["conditions"], list), ans
        return ans
    except Exception:
        logging.exception(f"Loading json failure: {ans}")
    return {"conditions": []}
```

**프롬프트 템플릿**: [rag/prompts/meta_filter.md](rag/prompts/meta_filter.md#L1-L40)

```markdown
You are a metadata filtering condition generator. Analyze the user's question and available document metadata to output a JSON array of filter objects. Follow these rules:

1. **Metadata Structure**: 
   - Metadata is provided as JSON where keys are attribute names (e.g., "color"), and values are objects mapping attribute values to document IDs.
   - Example: 
     {
       "color": {"red": ["doc1"], "blue": ["doc2"]},
       "listing_date": {"2025-07-11": ["doc1"], "2025-08-01": ["doc2"]}
     }

2. **Output Requirements**:
   - Always output a JSON dictionary with only 2 keys: 'conditions'(filter objects) and 'logic' between the conditions ('and' or 'or').
   - Each filter object in conditions must have:
        "key": (metadata attribute name),
        "value": (string value to compare),
        "op": (operator from allowed list)
   - Logic between all the conditions: 'and'(Intersection of results for each condition) / 'or' (union of results for all conditions)
```

**출력 예시**:
```json
{
  "conditions": [
    {
      "key": "author",
      "op": "=",
      "value": "Alice"
    },
    {
      "key": "publication_date",
      "op": ">",
      "value": "2024-01-01"
    }
  ],
  "logic": "and"
}
```

#### Mode 3: Semi-Automatic (반자동)
**프론트엔드**: [web/src/components/metadata-filter/metadata-semi-auto-fields.tsx](web/src/components/metadata-filter/metadata-semi-auto-fields.tsx#L1-L80)

```tsx
export function MetadataSemiAutoFields({ kbIds, prefix = '' }: {
  kbIds: string[];
  prefix?: string;
}) {
  const { t } = useTranslation();
  const form = useFormContext();
  const name = prefix + 'meta_data_filter.semi_auto';
  const metadata = useFetchKnowledgeMetadata(kbIds);  // KB의 메타데이터 키 목록 가져오기

  const { fields, remove, append } = useFieldArray({
    name,
    control: form.control,
  });

  const add = useCallback((key: string) => () => {
    append(key);  // 사용자가 선택한 필드만 추가
  }, [append]);

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <FormLabel>{t('metadataKeys')}</FormLabel>
        <DropdownMenu>
          <DropdownMenuTrigger>
            <Button variant={'ghost'} type="button">
              <Plus />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="max-h-[300px] !overflow-y-auto scrollbar-auto">
            {/* 사용 가능한 메타데이터 키 목록 표시 */}
            {Object.keys(metadata.data).map((key, idx) => (
              <DropdownMenuItem key={idx} onClick={add(key)}>
                {key}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {/* 선택된 필드 표시 */}
      {fields.map((field, index) => (
        <div key={field.id} className="flex items-center gap-2">
          <Input value={field.value} disabled />
          <Button variant={'ghost'} onClick={() => remove(index)}>
            <X />
          </Button>
        </div>
      ))}
    </section>
  );
}
```

**백엔드**: [common/metadata_utils.py](common/metadata_utils.py#L147-L157)

```python
elif method == "semi_auto":
    selected_keys = meta_data_filter.get("semi_auto", [])  # 사용자가 선택한 필드
    if selected_keys:
        # 선택된 필드만 필터링
        filtered_metas = {key: metas[key] for key in selected_keys if key in metas}
        if filtered_metas:
            # 선택된 필드 내에서 LLM이 자동으로 조건 생성
            filters: dict = await gen_meta_filter(chat_mdl, filtered_metas, question)
            doc_ids.extend(meta_filter(metas, filters["conditions"], filters.get("logic", "and")))
            if not doc_ids:
                return None
```

**동작 방식**:
1. 사용자: "author" 필드만 선택
2. LLM: 사용자 질문을 분석하여 "author"에 대한 조건만 생성
3. 예: `{"key": "author", "op": "=", "value": "Alice"}`

#### Mode 4: Manual (수동)
**프론트엔드**: [web/src/components/metadata-filter/metadata-filter-conditions.tsx](web/src/components/metadata-filter/metadata-filter-conditions.tsx#L27-L180)

```tsx
export function MetadataFilterConditions({ kbIds, prefix = '', canReference }: {
  kbIds: string[];
  prefix?: string;
  canReference?: boolean;
}) {
  const form = useFormContext();
  const name = prefix + 'meta_data_filter.manual';
  const logic = prefix + 'meta_data_filter.logic';
  const metadata = useFetchKnowledgeMetadata(kbIds);

  const switchOperatorOptions = useBuildSwitchOperatorOptions();

  const { fields, remove, append } = useFieldArray({
    name,
    control: form.control,
  });

  const add = useCallback((key: string) => () => {
    if (fields.length === 1) {
      form.setValue(logic, SwitchLogicOperator.And);
    }
    append({
      key,  // 필드 이름
      value: '',  // 비교할 값
      op: SwitchOperatorOptions[0].value,  // 연산자 (=, ≠, >, <, etc.)
    });
  }, [append, fields.length, form, logic]);

  return (
    <section className="flex flex-col gap-2">
      {/* 조건 추가 버튼 */}
      <div className="flex items-center justify-between">
        <FormLabel>{t('chat.conditions')}</FormLabel>
        <DropdownMenu>
          <DropdownMenuTrigger>
            <Button variant={'ghost'} type="button"><Plus /></Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            {Object.keys(metadata.data).map((key, idx) => (
              <DropdownMenuItem key={idx} onClick={add(key)}>
                {key}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      
      {/* 논리 연산자 선택 (AND/OR) */}
      <section className="flex">
        {fields.length > 1 && (
          <LogicalOperator logic={logic} />
        )}
      </section>
      
      {/* 조건 카드들 */}
      {fields.map((field, index) => (
        <ConditionCards fieldName={`${name}.${index}`} index={index} key={field.id} />
      ))}
    </section>
  );
}
```

**조건 카드 구조**:
```tsx
function ConditionCards({ fieldName, index }: { fieldName: string; index: number }) {
  return (
    <div className="flex gap-2">
      <Card className="flex-1">
        <CardHeader>
          {/* 필드 이름 표시 */}
          <FormField control={form.control} name={`${name}.${index}.key`} render={({ field }) => (
            <FormItem><FormControl><Input {...field} disabled /></FormControl></FormItem>
          )} />
        </CardHeader>
        <section className="p-2 flex justify-between">
          {/* 값 입력 */}
          <FormField control={form.control} name={fieldName} render={({ field }) => (
            <FormItem className="flex-1">
              <FormControl><Input {...field} placeholder={t('common.pleaseInput')} /></FormControl>
            </FormItem>
          )} />
          {/* 연산자 선택 */}
          <FormField control={form.control} name={`${name}.${index}.op`} render={({ field }) => (
            <FormItem>
              <FormControl>
                <RAGFlowSelect {...field} options={switchOperatorOptions} />
              </FormControl>
            </FormItem>
          )} />
        </section>
      </Card>
      <Button variant={'ghost'} onClick={() => remove(index)}><X /></Button>
    </div>
  );
}
```

**백엔드**: [common/metadata_utils.py](common/metadata_utils.py#L158-L164)

```python
elif method == "manual":
    filters = meta_data_filter.get("manual", [])
    # manual_value_resolver: 값 변환 함수 (예: 변수 참조 처리)
    if manual_value_resolver:
        filters = [manual_value_resolver(flt) for flt in filters]
    doc_ids.extend(meta_filter(metas, filters, meta_data_filter.get("logic", "and")))
    # 조건은 있지만 결과가 없으면 빈 결과 반환
    if filters and not doc_ids:
        doc_ids = ["-999"]
```

**조건 예시**:
```json
{
  "method": "manual",
  "logic": "and",
  "manual": [
    {
      "key": "author",
      "op": "=",
      "value": "Alice"
    },
    {
      "key": "publication_date",
      "op": ">",
      "value": "2024-01-01"
    }
  ]
}
```

### 2.3 메타데이터 필터 연산자

**지원되는 연산자**: [common/metadata_utils.py](common/metadata_utils.py#L46-L118)

```python
def meta_filter(metas: dict, filters: list[dict], logic: str = "and"):
    doc_ids = set([])

    def filter_out(v2docs, operator, value):
        ids = []
        for input, docids in v2docs.items():
            # 숫자 비교 연산자는 자동 형변환
            if operator in ["=", "≠", ">", "<", "≥", "≤"]:
                try:
                    if isinstance(input, list):
                        input = input[0]
                    input = ast.literal_eval(input)
                    value = ast.literal_eval(value)
                except Exception:
                    pass
            
            # 문자열은 소문자로 변환
            if isinstance(input, str):
                input = input.lower()
            if isinstance(value, str):
                value = value.lower()

            matched = False
            try:
                if operator == "contains":
                    matched = input in value if not isinstance(input, list) else all(i in value for i in input)
                elif operator == "not contains":
                    matched = input not in value if not isinstance(input, list) else all(i not in value for i in input)
                elif operator == "in":
                    matched = input in value if not isinstance(input, list) else all(i in value for i in input)
                elif operator == "not in":
                    matched = input not in value if not isinstance(input, list) else all(i not in value for i in input)
                elif operator == "start with":
                    matched = str(input).lower().startswith(str(value).lower())
                elif operator == "end with":
                    matched = str(input).lower().endswith(str(value).lower())
                elif operator == "empty":
                    matched = not input
                elif operator == "not empty":
                    matched = bool(input)
                elif operator == "=":
                    matched = input == value
                elif operator == "≠":
                    matched = input != value
                elif operator == ">":
                    matched = input > value
                elif operator == "<":
                    matched = input < value
                elif operator == "≥":
                    matched = input >= value
                elif operator == "≤":
                    matched = input <= value
            except Exception as e:
                logging.exception(e)
                matched = False

            if matched:
                ids.extend(docids)
        
        return ids

    # 모든 필터 조건 적용
    for k, v2docs in metas.items():
        for f in filters:
            if k != f["key"]:
                continue
            ids = filter_out(v2docs, f["op"], f["value"])
            if not doc_ids:
                doc_ids = set(ids)
            else:
                if logic == "and":
                    doc_ids = doc_ids & set(ids)  # 교집합
                else:
                    doc_ids = doc_ids | set(ids)  # 합집합
            if not doc_ids:
                return []
    
    return list(doc_ids)
```

### 2.4 메타데이터 조회 API

**프론트엔드**: [web/src/hooks/use-knowledge-request.ts](web/src/hooks/use-knowledge-request.ts#L308-L324)

```tsx
export function useFetchKnowledgeMetadata(kbIds: string[] = []) {
  const { data, isFetching: loading } = useQuery<
    Record<string, Record<string, string[]>>
  >({
    queryKey: [KnowledgeApiAction.FetchMetadata, kbIds],
    initialData: {},
    enabled: kbIds.length > 0,
    gcTime: 0,
    queryFn: async () => {
      // KB의 모든 메타데이터 조회
      const { data } = await kbService.getMeta({ kb_ids: kbIds.join(',') });
      return data?.data ?? {};
    },
  });

  return { data, loading };
}
```

**백엔드**: [api/db/services/document_service.py](api/db/services/document_service.py#L710-L738)

```python
@classmethod
@DB.connection_context()
def get_meta_by_kbs(cls, kb_ids):
    """
    Legacy metadata aggregator (backward-compatible).
    - Does NOT expand list values and a list is kept as one string key.
      Example: {"tags": ["foo","bar"]} -> meta["tags"]["['foo', 'bar']"] = [doc_id]
    - Expects meta_fields is a dict.
    Use when existing callers rely on the old list-as-string semantics.
    """
    fields = [cls.model.id, cls.model.meta_fields]
    meta = {}
    for r in cls.model.select(*fields).where(cls.model.kb_id.in_(kb_ids)):
        doc_id = r.id
        for k, v in r.meta_fields.items():
            if k not in meta:
                meta[k] = {}
            if not isinstance(v, list):
                v = [v]
            for vv in v:
                if vv not in meta[k]:
                    if isinstance(vv, list) or isinstance(vv, dict):
                        continue
                    meta[k][vv] = []
                meta[k][vv].append(doc_id)
    return meta
```

**반환 구조**:
```python
{
  "author": {
    "Alice": ["doc1", "doc3"],
    "Bob": ["doc2", "doc4"]
  },
  "category": {
    "research": ["doc1", "doc2"],
    "manual": ["doc3", "doc4"]
  },
  "publication_date": {
    "2025-01-15": ["doc1"],
    "2025-02-01": ["doc2", "doc3"]
  }
}
```

---

## 3. 데이터 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│ 1단계: Dataset Configuration (파싱 시)                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ 사용자: Metadata Schema 정의     │
        │ - field: "author"                │
        │ - type: "string"                 │
        │ - description: "작가 이름"        │
        │ - enum: ["Alice", "Bob"]  (선택) │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ parser_config.metadata에 저장    │
        │ parser_config.enable_metadata=true│
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ 문서 업로드 & 파싱               │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ task_executor.py                 │
        │ - 청크별로 gen_metadata() 호출   │
        │ - LLM이 스키마 기반 추출         │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ Document.meta_fields에 저장      │
        │ {"author": "Alice",              │
        │  "category": "research"}         │
        └──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2단계: Chat Configuration (검색 시)                             │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ useFetchKnowledgeMetadata()      │
        │ - KB의 모든 meta_fields 조회     │
        │ - 필드별 값 목록 집계            │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ 사용자: 필터링 모드 선택         │
        │ - Disabled: 필터링 안 함         │
        │ - Auto: LLM이 자동 생성          │
        │ - Semi-Auto: 필드 선택 → LLM    │
        │ - Manual: 수동 조건 입력         │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ 사용자 질문 입력                 │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ apply_meta_data_filter()         │
        │ - Auto: gen_meta_filter()        │
        │ - Semi: 선택 필드로 gen_meta_filter()│
        │ - Manual: 직접 meta_filter()     │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ meta_filter()                    │
        │ - 조건 평가                      │
        │ - doc_ids 필터링                 │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ retriever.retrieval()            │
        │ - 필터링된 doc_ids로 검색        │
        └──────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ 검색 결과 반환                   │
        └──────────────────────────────────┘
```

---

## 4. 사용 예시

### 4.1 시나리오: 연구 논문 검색

#### Step 1: Dataset 설정

**메타데이터 스키마 정의**:
```json
{
  "type": "object",
  "properties": {
    "author": {
      "type": "string",
      "description": "논문 저자",
      "enum": ["Alice", "Bob", "Charlie"]
    },
    "publication_year": {
      "type": "int",
      "description": "출판 연도"
    },
    "topic": {
      "type": "string",
      "description": "논문 주제 (AI, ML, NLP 등)"
    },
    "peer_reviewed": {
      "type": "bool",
      "description": "동료 평가 여부"
    }
  }
}
```

#### Step 2: 문서 파싱

**파싱 결과 (Document.meta_fields)**:
```json
// doc1.pdf
{
  "author": "Alice",
  "publication_year": 2023,
  "topic": "RAG",
  "peer_reviewed": true
}

// doc2.pdf
{
  "author": "Bob",
  "publication_year": 2024,
  "topic": "LLM",
  "peer_reviewed": true
}

// doc3.pdf
{
  "author": "Charlie",
  "publication_year": 2022,
  "topic": "NLP",
  "peer_reviewed": false
}
```

#### Step 3: Chat에서 필터링

**시나리오 A: Auto Mode**

사용자 질문: "Alice가 쓴 최신 RAG 논문에 대해 알려줘"

LLM 자동 생성 필터:
```json
{
  "conditions": [
    {"key": "author", "op": "=", "value": "Alice"},
    {"key": "topic", "op": "=", "value": "RAG"}
  ],
  "logic": "and"
}
```

결과: doc1.pdf만 검색 대상

**시나리오 B: Semi-Auto Mode**

사용자: "peer_reviewed" 필드 선택

사용자 질문: "동료 평가를 받은 논문 찾아줘"

LLM 생성 필터 (선택된 필드 내에서만):
```json
{
  "conditions": [
    {"key": "peer_reviewed", "op": "=", "value": "true"}
  ],
  "logic": "and"
}
```

결과: doc1.pdf, doc2.pdf 검색 대상

**시나리오 C: Manual Mode**

사용자가 직접 설정:
```json
{
  "method": "manual",
  "logic": "and",
  "manual": [
    {"key": "publication_year", "op": "≥", "value": "2023"},
    {"key": "peer_reviewed", "op": "=", "value": "true"}
  ]
}
```

결과: doc1.pdf, doc2.pdf 검색 대상

---

## 4.2 시나리오: 여러 Dataset 사용 시 특정 Dataset만 검색

### 문제 상황

**Chat 설정**:
- Knowledge Base 선택: Dataset A (의료 연구), Dataset B (의료 매뉴얼)
- 두 데이터셋 모두 `category="medical"` 메타데이터 가짐
- 사용자 질문: "의료 연구 논문에서 당뇨병 치료법 찾아줘"

**문제**: 
메타데이터 필터링은 `Document.meta_fields`만 확인하므로, **Dataset 정보로는 필터링 불가**

### 해결책 1: Dataset 구분 메타데이터 추가

#### Step 1: 각 Dataset에 고유 메타데이터 스키마 설정

**Dataset A (Medical_Research)**:
```json
{
  "properties": {
    "category": {"type": "string", "enum": ["medical"]},
    "source_type": {"type": "string", "enum": ["research_paper"]},
    "dataset_name": {"type": "string"}
  }
}
```

**Dataset B (Medical_Manuals)**:
```json
{
  "properties": {
    "category": {"type": "string", "enum": ["medical"]},
    "source_type": {"type": "string", "enum": ["manual"]},
    "dataset_name": {"type": "string"}
  }
}
```

#### Step 2: 파싱 후 SDK로 dataset_name 일괄 설정

```python
from ragflow_sdk import RAGFlow

rag = RAGFlow(api_key="YOUR_API_KEY", base_url="http://localhost")

# Dataset A의 모든 문서에 source_type 설정
dataset_a = rag.get_dataset(name="Medical_Research")
for doc in dataset_a.list_documents():
    current_meta = doc.meta_fields or {}
    current_meta["source_type"] = "research_paper"
    current_meta["dataset_name"] = "Medical_Research"
    doc.update({"meta_fields": current_meta})

# Dataset B의 모든 문서에 source_type 설정
dataset_b = rag.get_dataset(name="Medical_Manuals")
for doc in dataset_b.list_documents():
    current_meta = doc.meta_fields or {}
    current_meta["source_type"] = "manual"
    current_meta["dataset_name"] = "Medical_Manuals"
    doc.update({"meta_fields": current_meta})
```

#### Step 3: Chat에서 필터링

**Auto Mode 예시**:

사용자 질문: "의료 연구 논문에서 당뇨병 치료법 찾아줘"

LLM이 "연구 논문"이라는 키워드를 인식하여 자동 필터 생성:
```json
{
  "conditions": [
    {"key": "category", "op": "=", "value": "medical"},
    {"key": "source_type", "op": "=", "value": "research_paper"}
  ],
  "logic": "and"
}
```
→ Dataset A의 문서만 검색

**Semi-Auto Mode 예시**:

사용자가 `source_type` 필드 선택 후 질문: "매뉴얼 문서에서 당뇨병 관리 방법 알려줘"

LLM이 선택된 필드 내에서 조건 생성:
```json
{
  "conditions": [
    {"key": "source_type", "op": "=", "value": "manual"}
  ],
  "logic": "and"
}
```
→ Dataset B의 문서만 검색

**Manual Mode 예시**:

사용자가 직접 Dataset 지정:
```json
{
  "method": "manual",
  "logic": "and",
  "manual": [
    {"key": "dataset_name", "op": "=", "value": "Medical_Research"}
  ]
}
```
→ Dataset A의 문서만 검색

### 해결책 2: 각 Dataset의 문서에 고유한 메타데이터 값 사용

**Dataset A의 문서**:
```json
{
  "category": "medical_research",
  "institution": "Seoul_Hospital"
}
```

**Dataset B의 문서**:
```json
{
  "category": "medical_manual",
  "institution": "Healthcare_Training_Center"
}
```

→ `category` 값 자체를 다르게 설정하여 구분

### 해결책 3: LLM의 자연어 이해 활용 (Auto Mode)

**메타데이터 설계**:
```json
{
  "document_purpose": {
    "type": "string",
    "description": "Purpose of the document: 'research' (academic research), 'training' (educational/manual), 'clinical' (clinical guidelines)",
    "enum": ["research", "training", "clinical"]
  }
}
```

**사용자 질문과 LLM 필터 생성**:

| 사용자 질문 | LLM 생성 필터 | 검색 대상 |
|------------|-------------|----------|
| "최신 연구 결과 알려줘" | `{"key": "document_purpose", "op": "=", "value": "research"}` | Dataset A |
| "당뇨병 관리 매뉴얼 찾아줘" | `{"key": "document_purpose", "op": "=", "value": "training"}` | Dataset B |
| "임상 가이드라인 보여줘" | `{"key": "document_purpose", "op": "=", "value": "clinical"}` | Dataset C |

**LLM이 사용자 질문에서 추출하는 키워드**:
- "연구", "논문", "paper", "study" → `research`
- "매뉴얼", "가이드", "튜토리얼", "manual" → `training`
- "임상", "가이드라인", "프로토콜" → `clinical`

### 중요 포인트

1. **메타데이터 필터링은 Dataset 정보를 직접 사용하지 않음**
   - `kb_ids`는 검색 범위 지정에만 사용
   - 필터링은 `Document.meta_fields` 기반

2. **Dataset 구분을 위해서는 문서에 구분 메타데이터 추가 필요**
   - `source_type`, `dataset_name`, `document_purpose` 등
   - 파싱 후 SDK로 일괄 설정 가능

3. **Auto Mode의 LLM은 사용자 질문의 맥락을 이해**
   - "연구 논문"이라는 표현 → `source_type="research_paper"` 필터
   - "매뉴얼 문서"라는 표현 → `source_type="manual"` 필터
   - **단, 메타데이터 스키마에 해당 필드가 있어야 함**

4. **최선의 방법**:
   - **파싱 전**: 스키마에 Dataset 구분 필드 추가 (`source_type`, `dataset_name`)
   - **파싱 후**: SDK로 모든 문서에 Dataset별 값 일괄 설정
   - **Chat 사용 시**: Auto/Semi-Auto 모드에서 자연어로 구분 가능

### 실전 워크플로우

```python
# 1. Dataset 생성 및 메타데이터 스키마 설정
dataset_a = rag.create_dataset(name="Medical_Research")
dataset_a.update({
    "parser_config": {
        "metadata": {
            "properties": {
                "source_type": {"type": "string"},
                "category": {"type": "string"}
            }
        },
        "enable_metadata": False  # 자동 추출 비활성화
    }
})

# 2. 문서 업로드 및 파싱
dataset_a.upload_documents(["research1.pdf", "research2.pdf"])

# 3. 파싱 완료 후 메타데이터 일괄 설정
for doc in dataset_a.list_documents():
    doc.update({
        "meta_fields": {
            "source_type": "research_paper",
            "category": "medical",
            "dataset_name": "Medical_Research"
        }
    })

# 4. Chat 사용
# 사용자: "의료 연구 논문에서 찾아줘"
# → LLM이 자동으로 source_type="research_paper" 필터 생성
```

---

## 5. 핵심 포인트 요약

### 5.1 독립성
- **Dataset Configuration**: 파싱 시 메타데이터 **추출**
- **Chat Configuration**: 검색 시 메타데이터 **필터링**
- 두 단계는 독립적으로 동작 (한쪽만 사용 가능)

### 5.2 연결성
- Dataset에서 추출된 `meta_fields`가 Chat 필터링의 **데이터 소스**
- Chat의 필터링 UI는 `useFetchKnowledgeMetadata()`로 **가능한 필드 목록 조회**
- 필터링 결과는 `meta_filter()`로 **doc_ids 필터링**

### 5.3 유연성
- **Enum Mode**: Dataset 설정에서 값 제한 → LLM 추출 정확도 향상
- **Auto/Semi-Auto**: LLM이 자연어 질문을 필터 조건으로 변환
- **Manual**: 정확한 조건 설정 가능

### 5.4 성능 최적화
- **캐싱**: LLM 메타데이터 추출 결과 캐싱 (`get_llm_cache`, `set_llm_cache`)
- **병렬 처리**: 청크별 메타데이터 추출을 `asyncio.gather()`로 병렬 실행
- **인덱싱**: `meta_fields`는 JSONField로 저장되어 빠른 조회 가능

---

## 6. 코드 참조 요약

### Frontend
- **Metadata 설정 UI**: [web/src/pages/dataset/dataset-setting/configuration/common-item.tsx](web/src/pages/dataset/dataset-setting/configuration/common-item.tsx)
- **Metadata 필터 UI**: [web/src/components/metadata-filter/](web/src/components/metadata-filter/)
- **타입 정의**: [web/src/pages/dataset/components/metedata/interface.ts](web/src/pages/dataset/components/metedata/interface.ts)
- **메타데이터 조회 Hook**: [web/src/hooks/use-knowledge-request.ts](web/src/hooks/use-knowledge-request.ts)

### Backend
- **메타데이터 추출**: [rag/svr/task_executor.py#L401-L447](rag/svr/task_executor.py)
- **LLM 프롬프트**: [rag/prompts/generator.py](rag/prompts/generator.py)
- **메타데이터 필터링**: [common/metadata_utils.py](common/metadata_utils.py)
- **DB 모델**: [api/db/db_models.py#L790](api/db/db_models.py)
- **DB 서비스**: [api/db/services/document_service.py](api/db/services/document_service.py)

### Prompts
- **메타데이터 추출**: [rag/prompts/meta_data.md](rag/prompts/meta_data.md)
- **필터 조건 생성**: [rag/prompts/meta_filter.md](rag/prompts/meta_filter.md)
