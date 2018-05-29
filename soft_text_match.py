import difflib as dl
import nltk


# todo: issues to resolve
# todo: more than one with perfect match
# todo: not found


class SoftTextMatch():
    #
    def __init__(self, min_match_ratio=0.8, is_substring_match=True, is_stem=True):
        self.min_match_ratio = min_match_ratio
        self.is_substring_match = is_substring_match
        self.is_stem = is_stem

    def __match_term__(self, curr_term, term, max_ratio):
        is_match = False
        #
        if curr_term == term:
            max_ratio = 1
            is_match = True
        else:
            if self.is_substring_match and ((curr_term in term) or (term in curr_term)):
                if min(len(term), len(curr_term)) < 4:
                    substring_match = float(len(curr_term))/len(term)
                    if substring_match > 1:
                        substring_match = 1/substring_match
                else:
                    substring_match = self.min_match_ratio
                #
                if max_ratio is not None:
                    max_ratio = max(max_ratio, substring_match)
                else:
                    max_ratio = substring_match
                if substring_match == max_ratio:
                    is_match = True
            #
            dl_obj = dl.SequenceMatcher(None, curr_term, term)
            curr_ratio = dl_obj.quick_ratio()
            if max_ratio is not None:
                max_ratio = max(max_ratio, curr_ratio)
            else:
                max_ratio = curr_ratio
                
            if curr_ratio == max_ratio:
                is_match = True
        #
        if is_match:
            return max_ratio

    def find_max_match_term(self, org_term, terms_list):
        #
        if self.is_stem:
            ls_obj = nltk.stem.LancasterStemmer()
            org_term_stem_upper = ls_obj.stem(org_term).upper()
        else:
            org_term_stem_upper = org_term.upper()
        #
        max_ratio = None
        max_match_term = None
        max_match_term_idx = None
        #
        # todo: consider case of multiple nodes with same name
        #
        for curr_term_idx, curr_term in enumerate(terms_list):
            #
            if self.is_stem:
                curr_term_stem_upper = ls_obj.stem(curr_term).upper()
            else:
                curr_term_stem_upper = curr_term.upper()
            #
            max_ratio1 = self.__match_term__(curr_term_stem_upper, org_term_stem_upper, max_ratio)
            #
            max_ratio2 = self.__match_term__(curr_term.upper(), org_term_stem_upper, max_ratio)
            #
            max_ratio3 = self.__match_term__(curr_term_stem_upper, org_term.upper(), max_ratio)
            #
            max_ratio4 = self.__match_term__(curr_term.upper(), org_term.upper(), max_ratio)
            #
            curr_max_ratio = max(max_ratio1, max_ratio2, max_ratio3, max_ratio4)
            if max_ratio is not None:
                max_ratio = max(curr_max_ratio, max_ratio)
            else:
                max_ratio = curr_max_ratio
            #
            if max_ratio == curr_max_ratio:
                max_match_term_idx = curr_term_idx
            #
        if max_ratio is None or max_ratio < self.min_match_ratio:
            return None
        else:
            return terms_list[max_match_term_idx], max_match_term_idx, max_ratio


if __name__ == '__main__':
    # code for testing the above module
    terms = ['helllllo', 'hell?ji', 'helen']
    #
    curr_term = 'hello'
    stm_obj = SoftTextMatch()
    match_term, match_term_idx, max_ratio = stm_obj.find_max_match_term(curr_term, terms)
    print(match_term, match_term_idx, max_ratio)


