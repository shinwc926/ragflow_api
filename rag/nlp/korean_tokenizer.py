"""
kiwipiepy를 사용한 한국어 토크나이저
범용 한국어 토큰 추출기
"""

from kiwipiepy import Kiwi
import re
import sys
from pathlib import Path

class KoreanTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()
    
    def extract_tokens_with_position(self, text):
        """텍스트에서 내용어(명사, 동사, 형용사) 토큰과 위치 정보 추출"""
        try:
            analysis = self.kiwi.analyze(text)
            if analysis and analysis[0]:
                tokens_with_pos = []
                for token in analysis[0][0]:
                    # 명사류 + 접사 + 관형사 + 영어/숫자 포함
                    if token.tag in ('NNG', 'NNP', 'NNB', 'XSN', 'XPN', 'MM', 'SL', 'SN'):
                        # NNG = 일반명사 (예: 보험, 계약)
                        # NNP = 고유명사 (예: 피보험자)
                        # NNB = 의존명사 (예: 것, 수)
                        # XSN = 명사파생 접미사 (예: 저축성, 연계형의 "성", "형")
                        # XPN = 명사 접두사 (예: 주계약, 부보험의 "주", "부")
                        # MM = 관형사 (예: 주피보험자의 "주")
                        # SL = 영어
                        # SN = 숫자
                        tokens_with_pos.append({
                            'form': token.form,
                            'start': token.start,
                            'len': token.len,
                            'end': token.start + token.len
                        })
                return tokens_with_pos
        except Exception as e:
            print(f"토큰 추출 오류: {e}")
        return []
    
    def extract_tokens(self, text):
        """텍스트에서 내용어 토큰 추출 (기존 호환성 유지)"""
        tokens_with_pos = self.extract_tokens_with_position(text)
        return [token['form'] for token in tokens_with_pos]
    
    def _extract_content_tokens(self, analysis):
        """형태소 분석 결과에서 내용어 토큰 추출"""
        tokens = []
        for token in analysis[0][0]:
            if token.tag in ('NNG', 'NNP', 'NNB', 'XSN', 'XPN', 'MM', 'SL', 'SN', 'SW', 'NR'):
                # NNG = 일반명사, NNP = 고유명사, NNB = 의존명사
                # XSN = 명사파생 접미사, XPN = 명사 접두사, MM = 관형사
                # SL = 영어, SN = 숫자, SW = 기타기호
                # NR = 수사 (만, 억, 조, 첫째, 둘째 등)
                tokens.append({
                    'form': token.form,
                    'start': token.start,
                    'end': token.start + token.len,
                    'original_pos': token.start,
                    'tag': token.tag
                })
        return tokens
    
    def _should_group_tokens(self, current_tag, next_tag, has_space):
        """두 토큰을 그룹화할지 판단"""
        if not has_space:
            return True
        
        # 숫자 + 단위/기호 패턴 (예: "1,200 m", "0.03 %")
        if current_tag == 'SN' and next_tag in ('SL', 'NNB', 'SW'):
            return True
        
        # 기호 + 숫자 패턴 (예: "$ 100", "€ 50")
        if current_tag == 'SW' and next_tag == 'SN':
            return True
        
        return False
    
    def _has_boundary_symbol(self, text, start, end):
        """구간 사이에 그룹 분리 기호가 있는지 확인"""
        # 괄호 및 구분 기호
        boundary_symbols = {
            '(', ')', '[', ']', '{', '}',  # 괄호
            '「', '」', '『', '』', '〈', '〉',  # 한글/중국어 괄호
            '<', '>',                       # 꺾쇠
            '"', '"', "'", "'", '"', "'",   # 인용부호
            '/', '\\', '|',                 # 구분선
            '·', '※', '◦', '○', '●',      # 기호
            '：', '；',                      # 중국어 구두점
        }
        
        if start >= len(text) or end > len(text):
            return False
        
        between_text = text[start:end]
        return any(symbol in between_text for symbol in boundary_symbols)
    
    def _group_tokens_by_spacing(self, tokens, text):
        """띄어쓰기 및 구분 기호 기준으로 토큰 그룹화"""
        groups = []
        i = 0
        while i < len(tokens):
            current_group = [tokens[i]]
            current_end = tokens[i]['end']
            j = i + 1
            
            while j < len(tokens):
                next_token = tokens[j]
                between_text = text[current_end:next_token['start']]
                has_space = ' ' in between_text or '\t' in between_text or '\n' in between_text
                
                # 괄호나 구분 기호가 있으면 그룹 분리
                has_boundary = self._has_boundary_symbol(text, current_end, next_token['start'])
                
                if has_boundary:
                    # 구분 기호가 있으면 그룹 종료
                    break
                elif not has_space and (next_token['start'] - current_end <= 1):
                    current_group.append(next_token)
                    current_end = next_token['end']
                    j += 1
                elif has_space and len(current_group) > 0:
                    current_tag = current_group[-1]['tag']
                    next_tag = next_token['tag']
                    if self._should_group_tokens(current_tag, next_tag, has_space):
                        current_group.append(next_token)
                        current_end = next_token['end']
                        j += 1
                    else:
                        break
                else:
                    break
            
            groups.append(current_group)
            i = j if j > i else i + 1
        
        return groups
    
    def _process_affix_combinations(self, current_group, added):
        """접사(XSN/XPN/MM) 및 한글자 명사 결합 처리"""
        skip_indices = set()
        combined_forms = []
        
        k = 0
        while k < len(current_group):
            token_info = current_group[k]
            t = token_info['form']
            tag = token_info['tag']
            
            # XPN(접두사): 다음 명사와 결합
            if tag == 'XPN' and k + 1 < len(current_group):
                combined = t + current_group[k + 1]['form']
                if combined not in added and self.filter_relevant_tokens([combined]):
                    combined_forms.append(combined)
                    added.add(combined)
                skip_indices.add(k)
                skip_indices.add(k + 1)
            
            # XSN(접미사): 이전 명사와 결합
            elif tag == 'XSN' and k > 0:
                prev_token = current_group[k - 1]['form']
                combined = prev_token + t
                if combined not in added and self.filter_relevant_tokens([combined]):
                    combined_forms.append(combined)
                    added.add(combined)
                skip_indices.add(k - 1)
                skip_indices.add(k)
            
            # 한글자 명사가 명사 뒤에 위치하면 접미사로 간주
            elif tag == 'NNG' and len(t) == 1 and k > 0:
                prev_token = current_group[k - 1]['form']
                prev_tag = current_group[k - 1]['tag']
                
                is_suffix_pattern = False
                if prev_tag in ('NNG', 'NNP'):
                    if k == len(current_group) - 1:
                        is_suffix_pattern = True
                    elif k + 1 < len(current_group):
                        next_tag = current_group[k + 1]['tag']
                        if next_tag not in ('NNG', 'NNP', 'XSN'):
                            is_suffix_pattern = True
                
                if is_suffix_pattern:
                    combined = prev_token + t
                    if combined not in added and self.filter_relevant_tokens([combined]):
                        combined_forms.append(combined)
                        added.add(combined)
                    skip_indices.add(k - 1)
                    skip_indices.add(k)
            
            # MM(관형사): 다음 명사와 결합
            elif tag == 'MM' and k + 1 < len(current_group):
                combined = t + current_group[k + 1]['form']
                if combined not in added and self.filter_relevant_tokens([combined]):
                    combined_forms.append(combined)
                    added.add(combined)
                skip_indices.add(k)
                skip_indices.add(k + 1)
            
            k += 1
        
        return skip_indices, combined_forms
    
    def _build_combined_index_map(self, current_group, added):
        """결합 형태를 인덱스별로 매핑"""
        combined_by_index = {}
        
        for k in range(len(current_group)):
            token_info = current_group[k]
            tag = token_info['tag']
            t = token_info['form']
            
            # XPN(접두사): 현재 위치에 결합 형태 매핑
            if tag == 'XPN' and k + 1 < len(current_group):
                next_token = current_group[k + 1]['form']
                combined = t + next_token
                if combined in added:
                    combined_by_index[k] = combined
            
            # XSN(접미사): 이전 위치에 결합 형태 매핑
            elif tag == 'XSN' and k > 0:
                prev_token = current_group[k - 1]['form']
                combined = prev_token + t
                if combined in added:
                    combined_by_index[k - 1] = combined
            
            # MM(관형사): 현재 위치에 결합 형태 매핑
            elif tag == 'MM' and k + 1 < len(current_group):
                next_token = current_group[k + 1]['form']
                combined = t + next_token
                if combined in added:
                    combined_by_index[k] = combined
            
            # 한글자 접미사 패턴: 이전 위치에 결합 형태 매핑
            elif tag == 'NNG' and len(t) == 1 and k > 0:
                prev_token = current_group[k - 1]['form']
                combined = prev_token + t
                if combined in added:
                    combined_by_index[k - 1] = combined
        
        return combined_by_index
    
    def _add_tokens_with_order(self, current_group, skip_indices, combined_by_index, added, result_keywords):
        """토큰을 순서대로 추가 (개별 명사 다음에 결합 형태)"""
        for k, token_info in enumerate(current_group):
            if k in skip_indices:
                # skip된 인덱스라도 결합 형태가 있으면 추가
                if k in combined_by_index:
                    result_keywords.append(combined_by_index[k])
                continue
            
            t = token_info['form']
            tag = token_info['tag']
            if tag not in ('XSN', 'XPN', 'MM'):
                if t not in added and self.filter_relevant_tokens([t]):
                    result_keywords.append(t)
                    added.add(t)
            
            # 이 위치의 결합 형태가 있으면 바로 뒤에 추가
            if k in combined_by_index:
                result_keywords.append(combined_by_index[k])
    
    def extract_keywords_by_spacing(self, text):
        """형태소 분석 후 원문 띄어쓰기 기준으로 복합어/단일어 구분 및 분리된 명사도 복합명사 뒤에 추가"""
        try:
            analysis = self.kiwi.analyze(text)
            if not analysis or not analysis[0]:
                return []

            # 1. 내용어 토큰 추출
            tokens = self._extract_content_tokens(analysis)
            if not tokens:
                return []

            # 2. 띄어쓰기 기준으로 토큰 그룹화
            groups = self._group_tokens_by_spacing(tokens, text)
            
            # 3. 각 그룹별로 키워드 추출
            result_keywords = []
            for current_group in groups:
                group_tokens = [token['form'] for token in current_group]
                added = set()
                
                # 3.1 전체 복합어 생성
                if len(current_group) > 1:
                    compound_token = ''.join(group_tokens)
                    if self.filter_relevant_tokens([compound_token]):
                        result_keywords.append(compound_token)
                        added.add(compound_token)
                
                # 3.2 접사 결합 처리
                skip_indices, combined_forms = self._process_affix_combinations(current_group, added)
                
                # 3.3 결합 형태 인덱스 매핑
                combined_by_index = self._build_combined_index_map(current_group, added)
                
                # 3.4 토큰 순서대로 추가
                self._add_tokens_with_order(current_group, skip_indices, combined_by_index, added, result_keywords)

            # 중복 제거(순서 유지)
            seen = set()
            final_keywords = []
            for kw in result_keywords:
                if kw not in seen:
                    final_keywords.append(kw)
                    seen.add(kw)
            return final_keywords
        except Exception as e:
            print(f"토큰화 오류: {e}")
            return []
    
    
    def combine_compound_tokens(self, tokens):
        """기존 호환성을 위해 유지 (더 이상 사용하지 않음, 현재는 내용어 전체 처리)"""
        return [], tokens
    
    def filter_relevant_tokens(self, tokens):
        """유용한 내용어 토큰만 필터링 (명사, 동사, 형용사, 영어 단어/약어/숫자 포함)"""
        # 기본적인 불용어나 너무 일반적인 단어들 제외
        basic_stopwords = {
            # 대명사, 지시어, 의문사
            '것', '때', '경우', '후', '전', '중', '내', '외', '등', 
            '어떻게', '무엇', '언제', '어디서', '누구', '왜', '어떤', '몇',
            '이것', '저것', '그것', '여기', '저기', '거기', '이렇게', '저렇게', '그렇게',
            # 동사/형용사 관련 stopwords는 모두 삭제
        }
        
        filtered_tokens = []
        for token in tokens:
            # 불용어 체크
            if token in basic_stopwords:
                continue
            
            # 영어 단어/약어 체크 (영어가 포함된 경우 더 관대하게)
            has_english = any(c.isascii() and c.isalpha() for c in token)
            # 숫자 체크
            has_digit = any(c.isdigit() for c in token)
            
            is_relevant = False
            
            if has_english:
                # 영어가 포함된 경우: 1글자 이상이면 허용 (약어 포함)
                if len(token) >= 1 and any(c.isalpha() for c in token):
                    is_relevant = True
            
            # 숫자가 포함된 경우 (순수 숫자, 소수점, 또는 숫자+단위/기호)
            if has_digit:
                # 순수 숫자인 경우: 1자리 이상이면 허용
                if token.isdigit():
                    is_relevant = True
                # 소수점 숫자 (예: "0.03", "3.14")
                elif token.replace('.', '').replace(',', '').isdigit():
                    is_relevant = True
                # 숫자+문자 조합 (예: "20일", "30년", "100개", "0.03%")
                elif any(c.isdigit() for c in token):
                    is_relevant = True
            
            # 순수 한글인 경우: 기존 규칙 적용
            if not has_english and not has_digit:
                # 너무 짧은 단어 제외 (1글자)
                if len(token) > 1:
                    # 알파벳이 하나라도 있는 경우 허용
                    if any(c.isalpha() for c in token):
                        is_relevant = True
            
            # 유용성 체크 통과시 추가
            if is_relevant:
                filtered_tokens.append(token)
                
        return filtered_tokens
    
    def tokenize(self, text):
        """텍스트 토큰화 메인 함수 (extract_keywords에서 변경)"""
        #print(f"입력 텍스트: {text}")
        
        # 새로운 방식: 형태소 분석 + 원문 띄어쓰기 기준
        extracted_tokens = self.extract_keywords_by_spacing(text)
        #print(f"추출된 토큰: {extracted_tokens}")
        
        # extract_keywords_by_spacing에서 이미 필터링과 XSN/XPN/MM 처리가 완료되었으므로
        # 추가 필터링 없이 중복 제거만 수행
        # (기존에는 filter_relevant_tokens를 다시 호출했지만 이는 XSN/XPN/MM을 제거함)
        
        # 중복 제거 및 순서 유지
        unique_tokens = []
        for token in extracted_tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
        
        #print(f"최종 토큰: {unique_tokens}")
        return unique_tokens


def main():
    # 한국어 토크나이저 초기화
    tokenizer = KoreanTokenizer()
    
    # 테스트 질문들 (명사, 동사, 형용사, 영어 단어/약어/숫자+단위 포함)
    test_questions = [
        "주총 20일 전 공시는 어떤 부서에서 주관하며, 담당 부서는 무엇인가요?",
        "지체 없이 공시해야 하는 사항은 어떤 부서에서 주관하며, 관련 법규는 무엇인가요?",
        "IT 사업을 추진하는 과정에서 정보보호팀은 어떤 역할을 하나요?",
        "CCM 매뉴얼을 작성하는 주요 목적은 무엇인가요?",
        "30년 만기 상품을 개발할 때 고려해야 할 중요한 사항은?",
        "계약을 체결한 후 10일 이내에 해지할 수 있나요?",
        "신속하게 처리해야 하는 업무는 어떤 것들이 있나요?",
        "복잡한 법률 문제를 해결하는 방법은 무엇인가요?",
        "복잡한 법률 문제를 해결하는 먼저 비움직여야 하는 팀은 어디인가요?",
        "다중이용업소 의무보험에 대해 알려주세요.",
        "IT추진팀의 업무 범위는 어떻게 되나요?",
        "IT사업의 정의는 무엇인가요?",
        "연속승무시간은 무엇인가요?",
    ]
    
    print("=" * 80)
    print("kiwipiepy를 사용한 한국어 토크나이저 테스트 (명사/동사/형용사/영어/숫자)")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[테스트 {i}]")
        tokens = tokenizer.tokenize(question)
        print(f"최종 토큰: {', '.join(tokens)}")
        print("-" * 60)
    
    # 추가 테스트: 사용자 입력
    print("\n" + "=" * 80)
    print("대화형 테스트 (종료하려면 'quit' 입력)")
    print("=" * 80)
    
    while True:
        user_input = input("\n텍스트를 입력하세요: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
        if user_input:
            tokens = tokenizer.tokenize(user_input)
            print(f"추출된 토큰: {', '.join(tokens)}")


if __name__ == "__main__":
    main()
