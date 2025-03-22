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
LANG = [BAQ, CHN, CHN09, FRE, GER, HEB, HUN, KOR, POL, SWE, ENG,
        "en","ja","zh","ko","ar","fr", "de","sl"]
#        "bg","ca","cs","de","en","es", "fr","it","nl","no","ro","ru"]

'''
UD_LANG_TO_DIR = {
    "bg": "/UD_Bulgarian-BTB/bg_btb-ud-{split}.conllu",
    "ca": "/UD_Catalan-AnCora/ca_ancora-ud-{split}.conllu",
    "cs": "/UD_Czech-PDT/cs_pdt-ud-{split}.conllu",
    "de": "/UD_German-GSD/de_gsd-ud-{split}.conllu",
    "en": "/UD_English-EWT/en_ewt-ud-{split}.conllu",
    "es": "/UD_Spanish-AnCora/es_ancora-ud-{split}.conllu",
    "fr": "/UD_French-GSD/fr_gsd-ud-{split}.conllu",
    "it": "/UD_Italian-ISDT/it_isdt-ud-{split}.conllu",
    "nl": "/UD_Dutch-Alpino/nl_alpino-ud-{split}.conllu",
    "no": "/UD_Norwegian-Bokmaal/no_bokmaal-ud-{split}.conllu",
    "ro": "/UD_Romanian-RRT/ro_rrt-ud-{split}.conllu",
    "ru": "/UD_Russian-SynTagRus/ru_syntagrus-ud-{split}.conllu",
}
'''
LANG_TO_DIR = {
    "en": "/en_ewt-r2.15/en_ewt-ud-{split}.conllu",
    "ja": "/ja_gsd-r2.15/ja_gsd-ud-{split}.conllu",
    "zh": "/zh_gsdsimp-r2.15/zh_gsdsimp-ud-{split}.conllu",
    "ko": "/ko_gsd-r2.15/ko_gsd-ud-{split}.conllu",
    "ar": "/ar_padt-r2.15/ar_padt-ud-{split}.conllu",
    "fr": "/fr_gsd-r2.15/fr_gsd-ud-{split}.conllu",
    "de": "/de_gsd-r2.15/de_gsd-ud-{split}.conllu",
    "sl": "/sl_ssj-r2.15/sl_ssj-ud-{split}.conllu",
}
