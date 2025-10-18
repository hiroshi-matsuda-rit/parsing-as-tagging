DUMMY_LABEL = "Y|X"

DATA_PATH = "data/spmrl/"
DEP_DATA_PATH = "data/bht/"


TETRATAGGER = "tetra"
HEXATAGGER = "hexa"

TD_SR = "td-sr"
BU_SR = "bu-sr"
BERT = ["bert", "roberta", "robertaL"]
BERTCRF = ["bert+crf", "roberta+crf", "robertaL+crf"]
BERTLSTM = ["bert+lstm", "roberta+lstm", "robertaL+lstm"]

BAQ = "Basque"
CHN = "Chinese"
CHN09 = "Chinese-conll09"
FRE = "French"
GER = "German"
HEB = "Hebrew"
HUN = "Hungarian"
KOR = "Korean"
POL = "Polish"
SWE = "swedish"
ENG = "English"
"""
LANG = [BAQ, CHN, CHN09, FRE, GER, HEB, HUN, KOR, POL, SWE, ENG,
        "en","ja","zh","ko","ar","fr","de","sl","bg","ca","cs","es","it","nl","no","ro","ru","zt","e2"]
LANG_TO_DIR = {
    "en": "/en_ewt-r2.15/en_ewt-ud-{split}.conllu",
    "ja": "/ja_gsd-r2.15/ja_gsd-ud-{split}.conllu",
    "zh": "/zh_gsdsimp-r2.15/zh_gsdsimp-ud-{split}.conllu",
    "ko": "/ko_gsd-r2.15/ko_gsd-ud-{split}.conllu",
    "ar": "/ar_padt-r2.15/ar_padt-ud-{split}.conllu",
    "fr": "/fr_gsd-r2.15/fr_gsd-ud-{split}.conllu",
    "de": "/de_gsd-r2.15/de_gsd-ud-{split}.conllu",
    "sl": "/sl_ssj-r2.15/sl_ssj-ud-{split}.conllu",
    "bg": "/bg_btb-r2.15/bg_btb-ud-{split}.conllu",
    "ca": "/ca_ancora-r2.15/ca_ancora-ud-{split}.conllu",
    "cs": "/cs_pdt-r2.15/cs_pdt-ud-{split}.conllu",
    "es": "/es_ancora-r2.15/es_ancora-ud-{split}.conllu",
    "it": "/it_isdt-r2.15/it_isdt-ud-{split}.conllu",
    "nl": "/nl_alpino-r2.15/nl_alpino-ud-{split}.conllu",
    "no": "/no_bokmaal-r2.15/no_bokmaal-ud-{split}.conllu",
    "ro": "/ro_rrt-r2.15/ro_rrt-ud-{split}.conllu",
    "ru": "/ru_syntagrus-r2.15/ru_syntagrus-ud-{split}.conllu",
    "zt": "/zh_gsd-r2.15/zh_gsd-ud-{split}.conllu",
    "e2": "/en_ewt-r2.2/en_ewt-ud-{split}.conllu",
}
"""
LANG = [BAQ, CHN, CHN09, FRE, GER, HEB, HUN, KOR, POL, SWE, ENG,
        "en","ja","jb","fr","cs"]
LANG_TO_DIR = {
    "en": "/*en_ewt-r2.15/en_ewt-ud-{split}.conllu",
    "ja": "/*ja_gsd-r2.15/ja_gsd-ud-{split}.conllu",
    "jb": "/*ja_bccwj-r2.15/ja_bccwj-ud-{split}.conllu",
    "fr": "/*fr_gsd-r2.15/fr_gsd-ud-{split}.conllu",
    "cs": "/*cs_pdt-r2.15/cs_pdt-ud-{split}.conllu",
}
