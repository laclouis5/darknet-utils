from src import *
from pathlib import Path


if __name__ == "__main__":
    base_path = Path("~/Downloads/bipbip_database/")

    folders = [
        # Dataset 4.2
        'training_set/mais_haricot_feverole_pois/50/1',
        'training_set/mais_haricot_feverole_pois/50/2',
        'training_set/mais_haricot_feverole_pois/60/1',
        'training_set/mais_haricot_feverole_pois/60/2',
        'training_set/mais_haricot_feverole_pois/100/1',
        'training_set/mais_haricot_feverole_pois/100/2',
        'training_set/haricot_jeune',
        'training_set/carotte/2',
        'training_set/carotte/5',
        'training_set/mais/2',
        'training_set/mais/7',
        'training_set/mais/6',
        'validation_set',
        'training_set/2019-05-23_montoldre/mais/1',
        'training_set/2019-05-23_montoldre/mais/2',
        'training_set/2019-05-23_montoldre/mais/3',
        'training_set/2019-05-23_montoldre/mais/4',
        'training_set/2019-05-23_montoldre/haricot/1',
        'training_set/2019-05-23_montoldre/haricot/2',
        'training_set/2019-05-23_montoldre/haricot/3',
        'training_set/2019-05-23_montoldre/haricot/4',
        "training_set/2019-07-03_larrere/poireau/3",
        "training_set/2019-07-03_larrere/poireau/4",
        # Dataset 5.0
        "training_set/2019-09-25_montoldre/mais/1",
        "training_set/2019-09-25_montoldre/mais/2",
        "training_set/2019-09-25_montoldre/mais/3",
        "training_set/2019-09-25_montoldre/haricot",
        "training_set/2019-10-05_ctifl/mais_1",
        "training_set/2019-10-05_ctifl/mais_2",
        "training_set/2019-10-05_ctifl/haricot",
        # Dataset 6.0
        "haricot_debug_montoldre_2",
        "mais_debug_montoldre_2",
        # Database 6.1
        "training_set/2019-07-03_larrere/poireau/5",
        # Dataset 7.0
        "training_set/2020-10-01_ctifl/p0619_0928",
        "training_set/2020-10-01_ctifl/p0623_1241",
        "training_set/2020-10-01_ctifl/p0626_0816",
        "training_set/2020-10-01_ctifl/p0626_1420",
        "training_set/2020-10-01_ctifl/p0626_1423",
        "training_set/2020-10-01_ctifl/p0630_1420",
        "training_set/2020-10-01_ctifl/p0630_1427",
        "training_set/2020-10-01_ctifl/p0630_1428",
        "training_set/2020-10-01_ctifl/p0701_1308",
        "training_set/2020-10-01_ctifl/p0923_1627",
        "training_set/2020-10-01_ctifl/p0928_1042",
        # Dataset 8.0
        "training_set/2020-10-12_montoldre/bean_1",
        "training_set/2020-10-12_montoldre/bean_2",
        "training_set/2020-10-12_montoldre/bean_3",
        "training_set/2020-10-12_montoldre/maize_1",
        "training_set/2020-10-12_montoldre/maize_2",
        "training_set/2020-10-12_montoldre/maize_3",
        # Dataset 8.1
        "training_set/2020-10-12_montoldre/bean_4",
        "training_set/2020-10-12_montoldre/maize_4",
        # Dataset 9.0
        "training_set/2021-03-29_larrere/row_1",
    ]

    folders = [base_path / folder for folder in folders]
    no_obj_dir = base_path / "training_set/no_obj/"

    fr_to_en = {"mais": "maize", "haricot": "bean", "mais_tige": "stem_maize", "haricot_tige": "stem_bean"}
    en_to_nb = {l: str(i) for i, l in enumerate(fr_to_en.values())}
    labels = set(fr_to_en.keys())
    stem_labels = {l for l in labels if "tige" in l}

    resolve_xml_file_paths(folders)
    create_noobj_folder(no_obj_dir)

    annotations = parse_xml_folders(folders) \
        .filter(lambda box: box.label in labels) \
        .remove_empty() \
        .square_boxes(ratio=7.5/100, labels=stem_labels) \
        .map_labels(fr_to_en)
    annotations += parse_xml_folder(no_obj_dir)
    annotations.print_stats()

    create_yolo_trainval(annotations, exist_ok=True, shuffle=True)