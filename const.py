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
        "en","ja","zh","ko","ar","fr","de","sl","bg","ca","cs","es","it","nl","no","ro","ru","zt","jb"]
LANG_TO_DIR = {
    "en": "/English@en_ewt-r2.15/en_ewt-ud-{split}.conllu",
    "ja": "/Japanese@ja_gsd-r2.15/ja_gsd-ud-{split}.conllu",
    "fr": "/French@fr_gsd-r2.15/fr_gsd-ud-{split}.conllu",
    "cs": "/Czech@cs_pdt-r2.15/cs_pdt-ud-{split}.conllu",
    "zh": "/Chinese@zh_gsdsimp-r2.15/zh_gsdsimp-ud-{split}.conllu",
    "ko": "/Korean@ko_gsd-r2.15/ko_gsd-ud-{split}.conllu",
    "ar": "/Arabic@ar_padt-r2.15/ar_padt-ud-{split}.conllu",
    "de": "/German@de_gsd-r2.15/de_gsd-ud-{split}.conllu",
    "sl": "/Slovenian@sl_ssj-r2.15/sl_ssj-ud-{split}.conllu",
    "bg": "/Bulgarian@bg_btb-r2.15/bg_btb-ud-{split}.conllu",
    "ca": "/Croatian@ca_ancora-r2.15/ca_ancora-ud-{split}.conllu",
    "es": "/Spanish@es_ancora-r2.15/es_ancora-ud-{split}.conllu",
    "it": "/Italian@it_isdt-r2.15/it_isdt-ud-{split}.conllu",
    "nl": "/Dutch@nl_alpino-r2.15/nl_alpino-ud-{split}.conllu",
    "no": "/Norwegian@no_bokmaal-r2.15/no_bokmaal-ud-{split}.conllu",
    "ro": "/Romanian@ro_rrt-r2.15/ro_rrt-ud-{split}.conllu",
    "ru": "/Russian@ru_syntagrus-r2.15/ru_syntagrus-ud-{split}.conllu",
    "zt": "/Chinese@zh_gsd-r2.15/zh_gsd-ud-{split}.conllu",
    "jb": "/hexatagger@ja_bccwj-r2.15/ja_bccwj-ud-{split}.conllu",
}
