"""
База знаний с текстовыми описаниями типов повреждений автомобилей.

Содержит подробные описания различных типов повреждений для семантического анализа с помощью CLIP.
"""

# База описаний повреждений для CLIP анализа
DAMAGE_DESCRIPTIONS = {
    "scratch": [
        # Царапины
        "scratch on car body panel",
        "superficial scratch on automobile paint",
        "light surface scratch on vehicle exterior",
        "paint scratch on car door",
        "minor scratch damage on car surface",
        "fine linear scratch on automotive paint",
        "shallow scratch on vehicle body",
        "hairline scratch on car paintwork",
        "surface abrasion on automobile",
        "light paint damage scratch"
    ],
    
    "dent": [
        # Вмятины
        "dent on car door",
        "vehicle body dent damage",
        "deep dent on automobile panel",
        "car body indentation",
        "metal deformation dent",
        "impact dent on vehicle",
        "structural dent damage",
        "panel dent on car body",
        "automobile bodywork dent",
        "serious dent on vehicle surface"
    ],
    
    "crack": [
        # Трещины
        "crack in car windshield",
        "glass crack damage on vehicle",
        "structural crack on automobile",
        "windshield crack fracture",
        "body panel crack",
        "metal fatigue crack",
        "automotive glass crack",
        "structural damage crack",
        "severe crack on vehicle",
        "fracture crack damage"
    ],
    
    "shatter": [
        # Разбитое стекло
        "shattered windshield",
        "broken car window glass",
        "smashed automobile glass",
        "cracked and shattered window",
        "vehicle glass destruction",
        "windshield impact shatter",
        "glass fragmentation damage",
        "completely broken window",
        "shattered auto glass",
        "glass impact damage"
    ],
    
    "rust": [
        # Ржавчина
        "rust damage on car body",
        "metal corrosion on vehicle",
        "automobile rust oxidation",
        "surface rust on car panel",
        "structural rust damage",
        "metal degradation rust",
        "car body rust spots",
        "corrosion damage on automobile",
        "extensive rust on vehicle",
        "rusty metal surface"
    ],
    
    "broken_part": [
        # Сломанные детали
        "broken car headlight",
        "damaged automobile part",
        "broken vehicle component",
        "fractured car part",
        "shattered car light",
        "broken bumper piece",
        "separated car part",
        "damaged vehicle assembly",
        "broken exterior component",
        "fractured automobile part"
    ],
    
    "paint_damage": [
        # Повреждение краски
        "peeling car paint",
        "automobile paint damage",
        "chipped vehicle paint",
        "faded car paint finish",
        "paint blistering on car",
        "automotive paint degradation",
        "car paint bubbling",
        "surface paint damage",
        "paint flaking on vehicle",
        "overall paint deterioration"
    ],
    
    "smash": [
        # Сильные удары/разрушения
        "severe car impact damage",
        "major vehicle collision damage",
        "extensive smash damage",
        "crushed car body panel",
        "major impact deformation",
        "serious accident damage",
        "vehicle structural damage",
        "heavy impact destruction",
        "major collision deformation",
        "catastrophic vehicle damage"
    ],
    
    "glass_damage": [
        # Повреждение стекол
        "cracked car window",
        "damaged automobile glass",
        "fractured vehicle window",
        "glass surface damage",
        "windshield chip damage",
        "side window crack",
        "rear window damage",
        "glass impact mark",
        "window surface fracture",
        "automotive glass damage"
    ],
    
    "light_damage": [
        # Легкие повреждения
        "minor cosmetic damage",
        "light surface imperfection",
        "small paint chip",
        "tiny scratch mark",
        "cosmetic surface damage",
        "minimal vehicle damage",
        "light exterior blemish",
        "small surface defect",
        "minor aesthetic damage",
        "light cosmetic flaw"
    ]
}

# Расширенные описания для более точной классификации
DETAILED_DAMAGE_DESCRIPTIONS = {
    "scratch": [
        "superficial linear mark on car paint surface caused by sharp object",
        "fine hairline scratch on automobile exterior paintwork",
        "light surface abrasion on vehicle body panel",
        "shallow scratch damage on car door paint",
        "microscopic paint layer removal scratch",
        "clear coat scratch on automotive surface",
        "light keying damage on car paint",
        "surface level scratch without primer damage",
        "automobile paint surface scratch from debris",
        "fine automotive paint surface imperfection"
    ],
    
    "dent": [
        "metal panel deformation from blunt force impact",
        "car door indentation from physical impact",
        "vehicle body panel depression damage",
        "structural metal deformation on automobile",
        "deep panel dent from collision impact",
        "automobile bodywork structural damage",
        "metal surface depression on car body",
        "panel deformation from object impact",
        "structural integrity compromise dent",
        "automotive metal panel deformation"
    ],
    
    "crack": [
        "glass fracture line propagation from impact point",
        "structural material separation crack",
        "automobile glass stress fracture",
        "windshield impact crack formation",
        "material integrity failure crack",
        "glass structural failure line",
        "automotive safety glass crack",
        "laminated glass crack damage",
        "structural glass integrity loss",
        "glass material fracture propagation"
    ]
}

# Словарь соответствия классов YOLO и CLIP описаний
YOLO_TO_CLIP_MAPPING = {
    "scratch": "scratch",
    "dent": "dent", 
    "crack": "crack",
    "shatter": "shatter",
    "rust": "rust",
    "broken": "broken_part",
    "paint": "paint_damage",
    "smash": "smash",
    "glass": "glass_damage",
    "light": "light_damage"
}

# Все описания в одном списке для CLIP
ALL_DAMAGE_DESCRIPTIONS = []
DESCRIPTION_TO_CLASS = {}

for damage_class, descriptions in DAMAGE_DESCRIPTIONS.items():
    for desc in descriptions:
        ALL_DAMAGE_DESCRIPTIONS.append(desc)
        DESCRIPTION_TO_CLASS[desc] = damage_class

# Расширенный список с детальными описаниями
ALL_DETAILED_DESCRIPTIONS = []
DETAILED_DESCRIPTION_TO_CLASS = {}

for damage_class, descriptions in DETAILED_DAMAGE_DESCRIPTIONS.items():
    for desc in descriptions:
        ALL_DETAILED_DESCRIPTIONS.append(desc)
        DETAILED_DESCRIPTION_TO_CLASS[desc] = damage_class

# Комбинированный список
ALL_DESCRIPTIONS = ALL_DAMAGE_DESCRIPTIONS + ALL_DETAILED_DESCRIPTIONS
DESCRIPTION_TO_DAMAGE_CLASS = {**DESCRIPTION_TO_CLASS, **DETAILED_DESCRIPTION_TO_CLASS}

def get_class_descriptions(damage_type: str) -> list:
    """Получить все описания для конкретного класса повреждения"""
    descriptions = DAMAGE_DESCRIPTIONS.get(damage_type, [])
    detailed_descriptions = DETAILED_DAMAGE_DESCRIPTIONS.get(damage_type, [])
    return descriptions + detailed_descriptions

def get_all_descriptions() -> list:
    """Получить все описания повреждений"""
    return ALL_DESCRIPTIONS

def get_description_class(description: str) -> str:
    """Получить класс повреждения по описанию"""
    return DESCRIPTION_TO_DAMAGE_CLASS.get(description, "unknown")

def get_yolo_class_mapping(yolo_class: str) -> str:
    """Получить CLIP класс по YOLO классу"""
    return YOLO_TO_CLIP_MAPPING.get(yolo_class, "light_damage")

# Примеры для тестирования
TEST_DESCRIPTIONS = [
    "automobile with visible scratch damage on side panel",
    "vehicle showing significant dent damage from collision",
    "car windshield with extensive crack network",
    "completely shattered automotive glass surface",
    "severe rust corrosion on car body panels",
    "broken headlight assembly with glass fragments",
    "extensive paint damage with peeling and chipping",
    "major impact damage with structural deformation",
    "cracked window with star pattern fracture",
    "minor cosmetic scratch on car door"
]