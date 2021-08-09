from bs4 import BeautifulSoup as bs
from multiprocess import Pool



def _count_pages(xml_file):
    with open(xml_file, "r") as file:
        content = "".join(file.readlines())
        bs_content = bs(content, "lxml")
        return len(list(bs_content.find_all("div", {"class": "page"})))


def _count_text_boxes(xml_file):
    with open(xml_file, "r") as file:
        content = "".join(file.readlines())
        bs_content = bs(content, "lxml")
        Ps = [len(list(page.find_all("p"))) for page in bs_content.find_all("div", {"class": "page"})]
        return sum(Ps) / len(Ps)


def get_statistics():
    xmls = []
    for i in range(4984):
        xmls.append('/data/athar/ppt_generation/ppt_generation/slide_generator_data/data/'
                    + str(i) + '/slide.clean_tika.xml')

    pool = Pool(20)
    # result = pool.map(_count_pages, xmls)
    result = pool.map(_count_text_boxes, xmls)
    print(result[:10])
    print(sum(result) / len(result))
    pool.close()
    pool.join()


get_statistics()
