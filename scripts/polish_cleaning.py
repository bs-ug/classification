import re

pattern_photo = "Zdjęcie([\w .:;(),\-\"]*/ [\w /]* / " \
                "((© 2017 Associated Press)|(© Glowimages)|(123RF/PICSEL)|(AFP)|(Agencja FORUM)|" \
                "(Agencja TVN/-x-news)|(AP)|(Archiwum)|(East News)|(EPA)|(facebook.com)|(FORUM)|" \
                "(Forum | Video: Google Earth Pro)|(Getty Images)|(Getty Images Video: Reuters Archive)|" \
                "(INTERIA.PL)|(materiały prasowe)|(PAP)|(PAP/EPA)|(Policja)|(Reporter)|(Reuters)|(RMF)|" \
                "(Twitter)|(YouTube)))|([\w ]*/ Grafika RMF FM /)"
pattern_br = "(<br/> *)|(<br> *)|(\\\\<br\\\\> *)"
pattern_n = "\n *"
pattern_blank = "  +"
pattern_error = "�|\|"
pattern_author = "Autor: "
pattern_like = "Chcesz być na bieżąco"
pattern_see_more_video = "zobacz więcej wideo » [\w .:(),-/\"]* Video: [\w .:(),-/]*(" \
                         "(Ambasada USA w Polsce/Twitter)|(AP)|(AP, Reuters)|(APTN)|(Armada Argentina)|" \
                         "(Cable News? Network)|(CENTCOM)|(DARPA)|(DVIDSHUB)|(EBS)|(ENEX)|(EPA)|(espreso.tv)|" \
                         "(Espreso TV)|(European Union)|(facebook)|(Facebook)|(Fakty)|(Fakty o Świecie TVN24 BiS)|" \
                         "(Fakty w Południe)|(Fakty)|(Fakt?y TVN)|(FSA)|(Gazeta Wyborcza)|(Google Earth)|" \
                         "(Google Maps)|(gov.pl/edukacja)|(Human Rights Watch)|(IDF)|(Kanał 5)|(Kontakt 24)|(KPRM)|" \
                         "(kremlin.ru)|(MDAA?)|(mil.ru)|(Ministerstwo Informacji DRL)|(Ministerstwo Obrony)|" \
                         "(MO Rosji)|(MON)|(MSW)|(NASA)|(natochannel.tv)|(NATO)|(NATO Channel)|(NATO TV)|(OSP)|" \
                         "(PAP/EPA/)|(policja)|(Polskie Radio)|(Polskie Radio / Jedynka)|" \
                         "(Program III Polskiego Radia)|(Rada TV)|(Radio Swoboda)|(Radio Zet)|(reuters)|(Reuters)|" \
                         "(Reuters TV)|(RMF FM)|(sejm.gov.pl)|(Superwizjer TVN)|(twitter.com)|(Twitter)|(tvn24)|" \
                         "(TVN ?24)|(Ukraine Revolution)|(Unia Europejska)|(U.S. Air Force / public domain)|" \
                         "(US)|(USAF)|(USMC)|(US Army)|(US Navy)|([y|Y]ou[t|T]ube)|(Zdjęcia orga?nizatora))"
pattern_see_more_video_short = "zobacz więcej wideo » [\w .:(),-/]* Video: "
pattern_read_more = "[(A-ZĄĆĘŁŃÓŚŻŹ)|(0-9)]+[a-ż .,\-\"]*... czytaj dalej »"


def clean_text(content):
    if re.search(pattern_error, content):
        return ""
    result = re.search(pattern_br, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1]:]
        result = re.search(pattern_br, content)
    result = re.search(pattern_n, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1]:]
        result = re.search(pattern_n, content)
    result = re.search(pattern_see_more_video, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1] - 1:]
        result = re.search(pattern_see_more_video, content)
    result = re.search(pattern_see_more_video_short, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1] - 1:]
        result = re.search(pattern_see_more_video_short, content)
    result = re.search(pattern_read_more, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1] - 1:]
        result = re.search(pattern_read_more, content)
    result = re.search(pattern_author, content)
    while result:
        content = content[:result.regs[0][0]]
        result = re.search(pattern_author, content)
    result = re.search(pattern_like, content)
    while result:
        content = content[:result.regs[0][0]]
        result = re.search(pattern_like, content)
    result = re.search(pattern_blank, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1] - 1:]
        result = re.search(pattern_blank, content)
    result = re.search(pattern_photo, content)
    while result:
        content = content[:result.regs[0][0]] + content[result.regs[0][1] - 1:]
        result = re.search(pattern_photo, content)
    return content
