prompt = {
    "height" : "<image>\nTell the overall object's vertical height in json",
    "width" : "<image>\nTell the overall object's horizontal width in json",
    "depth" : "<image>\nTell the object's depth in json",
    "item_weight" : "<image>\nHow many grams is the object in image?",
    "maximum_weight_recommendation" : "<image>\nTell the full object's maximum recommended weight in json"
}

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

def build_prompt(entity):
    if entity in entity_unit_map:
        valid_units = ", ".join(entity_unit_map[entity])

        entext = entity.replace("_"," ")
        prompt = f"""<image>\nAnalyze the image carefully for all the text and numerical data. 
From your analysis, return the {entext} value of the object present in the image as integer in any of the following most suitable unit of measurement - {valid_units}.
Your answer MUST contain a value and a unit. For example, if item parameter is 47g or 12V, value is the number and unit is x.
Return your answer strictly in the following json format - {entity} : <value> , unit: <unit>"""
        
        # print(prompt)
        return prompt