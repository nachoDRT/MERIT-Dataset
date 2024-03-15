subjects = ["aleman", "biologia", "biologia_y_geologia", "ciencias_del_mundo", "ciencias_naturales", "dibujo_artistico", "dibujo_tecnico", "economia","educacion_etico_civica","educacion_fisica","educacion_plastica","filosofia","fisica_y_quimica","frances","generic","geografia_e_historia","griego", "historia","informatica","ingles","latin","lengua_castellana","matematicas","musica","religion","tecnologia"]
cursos = ["_3_de_la_eso", "_4_de_la_eso", "_1_de_bachillerato", "_2_de_bachillerato"]

text = ""

counter = 5

text += '{\n"other": "0",\n"3_de_la_eso": "1",\n"4_de_la_eso": "2",\n"1_de_bachillerato": "3",\n"2_de_bachillerato": "4",'

for subject in subjects:
    for curso in cursos:
        text += f'\n"{subject}{curso}": "{counter}",'
        counter += 1
    for curso in cursos:
        text += f'\n"{subject}{curso}_answer": "{counter}",'
        counter += 1

text += "\n}"
print(text)