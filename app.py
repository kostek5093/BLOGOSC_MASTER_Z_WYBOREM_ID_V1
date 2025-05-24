import streamlit as st
import pdfplumber
import re
import pandas as pd
import tempfile
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageDraw
import pytesseract
import os
import base64
import io
from fpdf import FPDF

# Ścieżka do tesseract.exe (zmień, jeśli masz inną)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- STYLE CSS DLA OKIEN UPLOADU ---
st.markdown("""
    <style>
    .upload-big {
    border: 4px solid #FFD600 !important; /* żółty */
    border-radius: 16px;
    padding: 24px 8px 8px 8px;
    background: #FFFDE7;
    margin-bottom: 16px;
    }
    .upload-big-green {
    border: 4px solid #43A047 !important; /* zielony */
    border-radius: 16px;
    padding: 24px 8px 8px 8px;
    background: #E8F5E9;
    margin-bottom: 16px;
    }
    .upload-small-blue {
    border: 3px solid #1976D2 !important; /* niebieski */
    border-radius: 12px;
    padding: 12px 8px 8px 8px;
    background: #E3F2FD;
    margin-bottom: 8px;
    }
    .upload-small-red {
    border: 3px solid #D32F2F !important; /* czerwony */
    border-radius: 12px;
    padding: 12px 8px 8px 8px;
    background: #FFEBEE;
    margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("BŁOGOŚĆ PRZEPIS MASTER")

# --- INICJALIZACJA ZMIENNYCH SESJI ---
if 'all_data_processed' not in st.session_state:
    st.session_state.all_data_processed = False
if 'all_ids' not in st.session_state:
    st.session_state.all_ids = []
if 'id_nazwa_list' not in st.session_state:
    st.session_state.id_nazwa_list = []
if 'przepisy_data' not in st.session_state:
    st.session_state.przepisy_data = {}
if 'gramatury_data' not in st.session_state:
    st.session_state.gramatury_data = {}
if 'df_sorted' not in st.session_state:
    st.session_state.df_sorted = None

# --- KOLOROWE OKNA UPLOADU ---
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="upload-big">', unsafe_allow_html=True)
    przepisy_file = st.file_uploader("Wgraj plik z przepisami (PDF)", type="pdf", key="przepisy")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="upload-big-green">', unsafe_allow_html=True)
    gramatury_file = st.file_uploader("Wgraj plik z gramaturami (PDF)", type="pdf", key="gramatury")
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="upload-small-blue">', unsafe_allow_html=True)
    zakupy_file = st.file_uploader("Lista zakupów (XLS/XLSX)", type=["xls", "xlsx"], key="zakupy")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="upload-small-red">', unsafe_allow_html=True)
    skladniki_file = st.file_uploader("Lista składników (XLS/XLSX)", type=["xls", "xlsx"], key="skladniki")
    st.markdown('</div>', unsafe_allow_html=True)

# --- DALEJ: LOGIKA APLIKACJI ---
if not (zakupy_file and gramatury_file and przepisy_file and skladniki_file):
    st.info("Aby zobaczyć sekcje: LISTA SKŁADNIKÓW, PRZEPISY i GRAMATURY, wrzuć wszystkie wymagane pliki.")
    st.stop()

# --- FUNKCJE POMOCNICZE ---
def extract_all_ids_from_gramatury(pdf_path):
    ids = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            found = re.findall(r'\(ID:\s*(\d+)\)', text)
            ids.update(found)
    return sorted(ids, key=lambda x: int(x))

def autocrop_image(image):
    gray = image.convert("L")
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return image

def find_lp_y(page):
    for word in page.extract_words():
        if re.search(r'\bLp\.', word['text']):
            return word['top']
    return None

def extract_multiline_field(lines, field):
    value_lines = []
    capture = False
    field_pattern = re.compile(rf"^{field}:", re.IGNORECASE)
    next_fields = [
        "dieta:", "wariant:", "posiłek:", "zamówiono:", "zamówiono (wpot):"
    ]
    indices = []
    for idx, line in enumerate(lines):
        if field_pattern.match(line.strip()):
            value_lines.append(line.split(":", 1)[1].strip())
            capture = True
            indices.append(idx)
            continue
        if capture:
            if any(line.strip().lower().startswith(f) for f in next_fields):
                break
            value_lines.append(line.strip())
            indices.append(idx)
    return " ".join(value_lines).strip(), set(indices)

def extract_recipe_info(page, danie_id):
    text = page.extract_text()
    if not text or f"(ID: {danie_id})" not in text:
        return None

    lines = text.split('\n')
    tytul = lines[1] if len(lines) > 1 else "Brak tytułu"
    tytul = re.sub(r'^przepisy?\s*', '', tytul, flags=re.IGNORECASE)

    dieta = wariant = posilek = dania = zamowiono = zamowiono_wpot = ""
    dania_indices = set()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("dieta:"):
            dieta = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("wariant:"):
            wariant = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("posiłek:"):
            posilek = line.split(":", 1)[1].strip()
        if line.strip().lower().startswith("zamówiono (wpot):"):
            zamowiono_wpot = line.split(":", 1)[1].strip()
        elif line.strip().lower().startswith("zamówiono:"):
            zamowiono = line.split(":", 1)[1].strip()
    dania, dania_indices = extract_multiline_field(lines, "dania")

    opis_lines = []
    opis_start = 2
    for i in range(opis_start, len(lines)):
        if re.search(r'\bLp\.', lines[i]):
            break
        if i in dania_indices:
            continue
        if any([
            lines[i].strip().lower().startswith("dieta:"),
            lines[i].strip().lower().startswith("wariant:"),
            lines[i].strip().lower().startswith("posiłek:"),
            lines[i].strip().lower().startswith("zamówiono:"),
            lines[i].strip().lower().startswith("zamówiono (wpot):")
        ]):
            continue
        opis_lines.append(lines[i])
    opis = "\n".join(opis_lines).strip()

    skladniki_start = None
    skladniki_end = None
    for i, line in enumerate(lines):
        if re.match(r'\s*Składniki', line, re.IGNORECASE):
            skladniki_start = i
        if skladniki_start is not None and line.strip().lower().startswith("razem"):
            skladniki_end = i
            break

    df = None
    if skladniki_start is not None and skladniki_end is not None:
        skladniki_lines = lines[skladniki_start:skladniki_end]
        header = re.split(r'\s{2,}|\t', skladniki_lines[0].strip())
        data = []
        for row in skladniki_lines[1:]:
            cols = re.split(r'\s{2,}|\t', row.strip())
            cols += [""] * (len(header) - len(cols))
            data.append(cols)
        df = pd.DataFrame(data, columns=header)

    return {
        "tytul": tytul,
        "opis": opis,
        "skladniki": df,
        "dieta": dieta,
        "wariant": wariant,
        "posilek": posilek,
        "dania": dania,
        "zamowiono": zamowiono,
        "zamowiono_wpot": zamowiono_wpot
    }

def extract_gramatura_info(pdf, danie_id):
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        pattern = r'\(ID: ?' + str(danie_id) + r'\)'
        if re.search(pattern, text):
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    match = re.search(r'\(ID: ?\d+\)\s*(.*)', line)
                    nazwa_dania = match.group(1).strip() if match else "Nazwa nieznana"
                    fragment = lines[i:i+5]
                    fragment = [l for l in fragment if not re.fullmatch(r'\d+\s*', l.strip()) and l.strip()]
                    return nazwa_dania, "\n".join(fragment)
    return "Nazwa nieznana", "Nie znaleziono gramatury dla tego ID."

def extract_gramatura_image_fragment_for_id(pdf_path, danie_id, margin=50):
    id_pattern = re.compile(rf"\b{re.escape(str(danie_id))}\b")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            if not id_pattern.search(text):
                continue
            y_start = None
            y_end = None
            for word in page.extract_words():
                if y_start is None and re.search(r'Przepisy/Składniki', word['text'], re.IGNORECASE):
                    y_start = word['top']
                if re.search(r'Pudełka', word['text'], re.IGNORECASE):
                    y_end = word['bottom']
            if y_start is not None and y_end is not None:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                img = images[0]
                pdf_height = page.height
                img_height = img.height
                y_start_px = int(y_start * img_height / pdf_height) - margin
                y_end_px = int(y_end * img_height / pdf_height) + margin
                y_start_px = max(0, y_start_px)
                y_end_px = min(img.height, y_end_px)
                cropped_img = img.crop((0, y_start_px, img.width, y_end_px))
                return cropped_img
    return None

def extract_przepis_image_fragment_for_page(pdf_path, page_num, margin_top=20, margin_bottom=50):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        lp_y = find_lp_y(page)
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        img = images[0]
        if lp_y is not None:
            pdf_height = page.height
            img_height = img.height
            y_px = int(lp_y * img_height / pdf_height) - margin_top
            y_px = max(0, y_px)
            bottom_px = min(img.height, img.height + margin_bottom)
            cropped_img = img.crop((0, y_px, img.width, bottom_px))
            cropped_img = autocrop_image(cropped_img)
            return cropped_img
        else:
            cropped_img = autocrop_image(img)
            return cropped_img

def extract_ids_from_dania(dania_text, main_id):
    found_ids = set(re.findall(r'\(ID:\s*(\d+)\)', dania_text))
    found_ids.discard(str(main_id))
    return list(found_ids)

def extract_ingredients_from_gramatura_image(image):
    import re
    text = pytesseract.image_to_string(image, lang='pol')
    lines = text.split('\n')
    ingredients = []
    blacklist = {'zamówiono', 'kt', 'ki', 'gramatury', 'gramatur', 'kcal'}
    substrings = ['office', 'standard', 'sport', 'wege', 'student', 'bez', 'pudełka', 'vege', 'zwykłe', 'eko', 'kg', 'vegan', 'kę', 'swe', 'kości', 'przepisy', 'uniwersalne', 'posiłków',]
    for line in lines:
        match = re.search(r'([a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s\-]+)[^\d]*(\d+[.,]?\d*)\s*(kg|g)?', line)
        if not match:
            match = re.search(r'([a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s\-]+)[^\d]*([.,]?\d+)\s*(kg|g)?', line)
        if match:
            name = match.group(1).strip()
            value = match.group(2)
            unit = match.group(3)
            name_lower = name.lower()
            if name_lower in blacklist:
                continue
            if any(sub in name_lower for sub in substrings):
                continue
            if len(name.replace(" ", "")) < 2:
                continue
            value = value.replace(',.', ',')
            value = value.replace('.,', ',')
            value = re.sub(r'^0[.,]+', '0,', value)
            if value.startswith(',') or value.startswith('.'):
                value = '0' + value
            value = value.replace(',', '.')
            try:
                weight = float(value)
                if unit == 'kg':
                    weight = weight * 1000
                ingredients.append((name, weight))
            except ValueError:
                print(f"Nie można sparsować liczby: {value} w linii: {line}")
                pass
        else:
            print("Niepasująca linia:", line)
    return ingredients

def calculate_pre_cooking_weights(ingredients, excel_df):
    results = []
    for name, weight_wpot in ingredients:
        wpot_percent = None
        for idx, row in excel_df.iterrows():
            excel_name = str(row['Nazwa']).lower() if pd.notnull(row['Nazwa']) else ""
            ingredient_name = name.lower()
            if (ingredient_name in excel_name or
                any(word in excel_name for word in ingredient_name.split() if len(word) > 3)):
                if len(row) >= 5 and pd.notnull(row.iloc[4]):
                    try:
                        wpot_percent = float(row.iloc[4])
                    except (ValueError, TypeError):
                        wpot_percent = None
                break
        if wpot_percent is not None:
            wpot_coef = wpot_percent / 100
            weight_before = weight_wpot / wpot_coef
            results.append({
                'name': name,
                'weight_wpot': weight_wpot,
                'weight_before': weight_before,
                'wpot_percent': wpot_percent,
                'found_coefficient': True
            })
        else:
            results.append({
                'name': name,
                'weight_wpot': weight_wpot,
                'found_coefficient': False
            })
    return results

def export_df_to_excel(df, sheet_name="Sheet1"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# --- NOWA FUNKCJA: podświetlanie największej masy/wagi ---
def highlight_max_weight_on_image(image):
    tsv = pytesseract.image_to_data(image, lang='pol', output_type=pytesseract.Output.DATAFRAME)
    tsv = tsv[tsv['text'].notnull() & (tsv['text'].str.strip() != '')]

    # Szukamy nagłówka kolumny z wagą/masą
    header_row = None
    weight_col = None
    possible_headers = ['masa', 'waga', 'waga [g]', 'masa [g]']
    for idx, row in tsv.iterrows():
        text = row['text'].strip().lower()
        for header in possible_headers:
            if header in text:
                header_row = row['line_num']
                weight_col = row['left']
                break
        if header_row is not None:
            break
    if header_row is None or weight_col is None:
        return image  # nie znaleziono kolumny

    # Zbieramy wartości pod nagłówkiem
    table_rows = tsv[tsv['line_num'] > header_row]
    max_val = -1
    max_box = None
    for idx, row in table_rows.iterrows():
        try:
            val = float(row['text'].replace(',', '.'))
            if val > max_val:
                max_val = val
                # RAMKA: powiększamy o 60% szerokości i 80% wysokości, przesuwamy w górę i w lewo
                margin_x = int(row['width'] * 0.6)
                margin_y = int(row['height'] * 0.8)
                left = max(0, row['left'] - margin_x)
                top = max(0, row['top'] - margin_y)
                right = row['left'] + row['width'] + margin_x
                bottom = row['top'] + row['height'] + margin_y
                max_box = (left, top, right, bottom)
        except:
            continue

    # Rysujemy prostokąt na największej masie/wadze
    if max_box:
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle(max_box, outline="red", width=8)
        return img_draw
    else:
        return image

# --- FUNKCJE EKSPORTU DO PDF ---
def export_all_recipes_pdf():
    pdf = FPDF()
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 16)

    for selected_id in st.session_state.all_ids:
        if selected_id in st.session_state.przepisy_data:
            for przepis in st.session_state.przepisy_data[selected_id]:
                pdf.add_page()
                pdf.set_font("DejaVu", "", 16)
                pdf.cell(0, 10, f"Przepis ID {selected_id} - {przepis['tytul']}", ln=True)
                pdf.ln(5)
                pdf.set_font("DejaVu", "", 12)
                pdf.multi_cell(0, 8, f"DIETA: {przepis['dieta']}\nWARIANT: {przepis['wariant']}\nPOSIŁEK: {przepis['posilek']}\nDANIA: {przepis['dania']}\nZAMÓWIONO: {przepis['zamowiono']}\nZAMÓWIONO (WpOT): {przepis['zamowiono_wpot']}\n\nOpis przygotowania:\n{przepis['opis']}\n")
                # Składniki (jeśli są)
                if przepis['skladniki'] is not None and not przepis['skladniki'].empty:
                    pdf.multi_cell(0, 8, "\nSKŁADNIKI DLA PRZEPISU:\n" + przepis['skladniki'].to_string(index=False))
                # Obraz (jeśli jest)
                if 'image' in przepis and przepis['image'] is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        img_path = tmp.name
                        przepis['image'].save(img_path, format='PNG')
                    pdf.add_page()
                    pdf.image(img_path, x=10, y=30, w=180)
                    os.unlink(img_path)
    return pdf.output(dest="S").encode("latin1")

def export_all_gramatury_pdf():
    pdf = FPDF()
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 16)

    for selected_id in st.session_state.all_ids:
        if selected_id in st.session_state.gramatury_data:
            gramatura_data = st.session_state.gramatury_data[selected_id]
            pdf.add_page()
            pdf.set_font("DejaVu", "", 16)
            pdf.cell(0, 10, f"Gramatura ID {selected_id} - {gramatura_data['nazwa']}", ln=True)
            pdf.ln(5)
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(0, 8, f"{gramatura_data['text']}\n\n")
            # Obraz (jeśli jest)
            if 'image' in gramatura_data and gramatura_data['image'] is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img_path = tmp.name
                    gramatura_data['image'].save(img_path, format='PNG')
                pdf.add_page()
                pdf.image(img_path, x=10, y=30, w=180)
                os.unlink(img_path)
    return pdf.output(dest="S").encode("latin1")

# --- PRZETWARZANIE DANYCH ---
# Jeśli dane nie zostały jeszcze przetworzone, przetwórz je
if not st.session_state.all_data_processed:
    with st.spinner("Przetwarzanie wszystkich danych... To może potrwać kilka minut."):
        # --- WCZYTYWANIE I PRZETWARZANIE PLIKÓW EXCEL ---
        try:
            ext_zakupy = os.path.splitext(zakupy_file.name)[1].lower()
            if ext_zakupy == ".xlsx":
                df_zakupy = pd.read_excel(zakupy_file, engine="openpyxl")
            elif ext_zakupy == ".xls":
                df_zakupy = pd.read_excel(zakupy_file, engine="xlrd")
            else:
                st.error("Nieobsługiwany format pliku z zakupami.")
                st.stop()

            ext_skladniki = os.path.splitext(skladniki_file.name)[1].lower()
            if ext_skladniki == ".xlsx":
                df_skladniki = pd.read_excel(skladniki_file, engine="openpyxl")
            elif ext_skladniki == ".xls":
                df_skladniki = pd.read_excel(skladniki_file, engine="xlrd")
            else:
                st.error("Nieobsługiwany format pliku ze składnikami.")
                st.stop()
        except Exception as e:
            st.error(f"Błąd wczytywania plików Excel: {e}")
            st.stop()

        # Wyciągamy kolumny C, D, E, N, O, P (indeksy 2, 3, 4, 13, 14, 15)
        expected_cols = [2, 3, 4, 13, 14, 15]
        if max(expected_cols) >= len(df_zakupy.columns):
            st.error("Plik z zakupami nie ma wymaganych kolumn (C, D, E, N, O, P).")
            st.stop()

        df_selected = df_zakupy.iloc[:, expected_cols].copy()
        df_selected.columns = [
            "SKŁADNIK", "KATEGORIA", "WAGA (netto) [g]", "DIETY", "DANIA", "PRZEPISY"
        ]

        # Dodajemy kolumnę "Waga (brutto WpOT) [g]"
        def calculate_brutto(row):
            skladnik = str(row["SKŁADNIK"]).strip().lower()
            waga_netto = row["WAGA (netto) [g]"]
            found = df_skladniki[df_skladniki.iloc[:, 1].astype(str).str.strip().str.lower() == skladnik]
            if not found.empty:
                try:
                    wpot_percent = float(found.iloc[0, 4])  # kolumna E (indeks 4)
                    if wpot_percent == 0:
                        return ""
                    wpot_coef = wpot_percent / 100
                    waga_brutto = float(waga_netto) / wpot_coef
                    return round(waga_brutto, 2)
                except Exception:
                    return ""
            return ""

        df_selected.insert(3, "Waga (brutto WpOT) [g]", df_selected.apply(calculate_brutto, axis=1))

        # Kolejność kategorii
        kategorie_kolejnosc = [
            "MIĘSO", "RYBY", "WARZYWA", "PRODUKTY ZBOŻOWE", "OWOCE", "NABIAŁ", "PRZYPRAWY"
        ]

        def get_category_priority(cat):
            if not isinstance(cat, str):
                return len(kategorie_kolejnosc) + 1
            cat_upper = cat.upper()
            for idx, wzorzec in enumerate(kategorie_kolejnosc):
                if wzorzec in cat_upper:
                    return idx
            return len(kategorie_kolejnosc) + 1

        df_selected['_sort'] = df_selected['KATEGORIA'].apply(get_category_priority)
        df_sorted = df_selected.sort_values(by=['_sort', 'KATEGORIA', 'SKŁADNIK']).drop(columns=['_sort'])

        # Zapisz posortowane dane do zmiennej sesji
        st.session_state.df_sorted = df_sorted

        # --- WCZYTAJ PDFY DO PLIKÓW TYMCZASOWYCH ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_przepisy:
            tmp_przepisy.write(przepisy_file.read())
            tmp_przepisy_path = tmp_przepisy.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_gramatury:
            tmp_gramatury.write(gramatury_file.read())
            tmp_gramatury_path = tmp_gramatury.name

        # Pobierz wszystkie ID
        st.session_state.all_ids = extract_all_ids_from_gramatury(tmp_gramatury_path)

        # Przygotuj listę ID z nazwami
        with pdfplumber.open(tmp_gramatury_path) as pdf_gramatury:
            id_nazwa_list = []
            for id_ in st.session_state.all_ids:
                nazwa, _ = extract_gramatura_info(pdf_gramatury, id_)
                id_nazwa_list.append(f"{id_} - {nazwa}")
            st.session_state.id_nazwa_list = id_nazwa_list

        # --- PRZETWARZANIE PRZEPISÓW ---
        with pdfplumber.open(tmp_przepisy_path) as pdf_przepisy:
            for danie_id in st.session_state.all_ids:
                przepisy = []
                for page_num, page in enumerate(pdf_przepisy.pages):
                    info = extract_recipe_info(page, danie_id)
                    if info:
                        info['page_num'] = page_num
                        # Dodaj obraz przepisu
                        cropped_img = extract_przepis_image_fragment_for_page(
                            tmp_przepisy_path,
                            page_num,
                            margin_top=20,
                            margin_bottom=50
                        )
                        # Podświetl największą masę/wagę
                        highlighted_img = highlight_max_weight_on_image(cropped_img)
                        info['image'] = highlighted_img
                        przepisy.append(info)
                if przepisy:
                    st.session_state.przepisy_data[danie_id] = przepisy

        # --- PRZETWARZANIE GRAMATUR ---
        with pdfplumber.open(tmp_gramatury_path) as pdf_gramatury:
            for danie_id in st.session_state.all_ids:
                nazwa_dania, gramatura_text = extract_gramatura_info(pdf_gramatury, danie_id)
                gramatura_img = extract_gramatura_image_fragment_for_id(tmp_gramatury_path, danie_id, margin=50)

                ingredients_data = []
                if gramatura_img is not None:
                    ingredients = extract_ingredients_from_gramatura_image(gramatura_img)
                    if ingredients:
                        ingredients_data = calculate_pre_cooking_weights(ingredients, df_skladniki)

                st.session_state.gramatury_data[danie_id] = {
                    'nazwa': nazwa_dania,
                    'text': gramatura_text,
                    'image': gramatura_img,
                    'ingredients_data': ingredients_data
                }

        # Usuń tymczasowe pliki
        os.unlink(tmp_przepisy_path)
        os.unlink(tmp_gramatury_path)

        # Oznacz dane jako przetworzone
        st.session_state.all_data_processed = True

# --- EKSPORT CAŁYCH SEKCJI DO PDF ---
st.header("EKSPORT DANYCH")

colA, colB = st.columns(2)
with colA:
    st.download_button(
        label="Pobierz wszystkie przepisy (PDF)",
        data=export_all_recipes_pdf(),
        file_name="wszystkie_przepisy.pdf",
        mime="application/pdf"
    )
with colB:
    st.download_button(
        label="Pobierz wszystkie gramatury (PDF)",
        data=export_all_gramatury_pdf(),
        file_name="wszystkie_gramatury.pdf",
        mime="application/pdf"
    )

# --- EKSPORT LISTY SKŁADNIKÓW (Excel) ---
st.subheader("Eksport listy składników")
excel_data = export_df_to_excel(st.session_state.df_sorted, "Lista_składników")
st.download_button(
    label="Pobierz listę składników (Excel)",
    data=excel_data,
    file_name="lista_skladnikow.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --- LISTA SKŁADNIKÓW ---
st.header("LISTA SKŁADNIKÓW")
st.markdown("### Lista składników (posortowana według kategorii):")
st.dataframe(st.session_state.df_sorted, use_container_width=True)

# --- WYBÓR PRZEPISÓW I GRAMATUR DO WYŚWIETLENIA ---
# Domyślnie zaznaczone wszystkie
selected_options = st.multiselect(
    "Wybierz przepisy/gramatury do wyświetlenia:",
    options=st.session_state.id_nazwa_list,
    default=st.session_state.id_nazwa_list
)

# Wyciągamy wybrane ID
selected_ids = [opt.split(" - ")[0] for opt in selected_options]

# --- PRZEPISY ---
st.header("PRZEPISY")

for selected_id in selected_ids:
    if selected_id in st.session_state.przepisy_data:
        for idx, przepis in enumerate(st.session_state.przepisy_data[selected_id], 1):
            st.markdown(
                f'<h3 style="color:#FFD600; margin-bottom: 0.5em;">Przepisy dla ID {selected_id} – {przepis["tytul"]}</h3>',
                unsafe_allow_html=True
            )
            st.markdown(f"**DIETA:** {przepis['dieta']}")
            st.markdown(f"**WARIANT:** {przepis['wariant']}")
            st.markdown(f"**POSIŁEK:** {przepis['posilek']}")
            st.markdown(f"**DANIA:** {przepis['dania']}")
            st.markdown(f"**ZAMÓWIONO:** {przepis['zamowiono']}")
            st.markdown(f"**ZAMÓWIONO (WpOT):** {przepis['zamowiono_wpot']}")

            if przepis['skladniki'] is not None and not przepis['skladniki'].empty:
                st.markdown("**SKŁADNIKI DLA PRZEPISU:**")
                st.dataframe(przepis['skladniki'], use_container_width=True)
            if przepis['opis']:
                st.markdown("**Opis przygotowania:**")
                st.write(przepis['opis'])

            st.markdown("**SKŁADNIKI DLA PRZEPISU (obraz):**")
            st.image(przepis['image'], use_container_width=True)
    else:
        st.warning(f"Nie znaleziono przepisów dla ID {selected_id}.")

# --- GRAMATURY ---
st.header("GRAMATURY")

for selected_id in selected_ids:
    if selected_id in st.session_state.gramatury_data:
        gramatura_data = st.session_state.gramatury_data[selected_id]

        st.markdown(
            f'<h3 style="color:#43A047; margin-bottom: 0.5em;">Gramatura dla ID {selected_id} – {gramatura_data["nazwa"]}</h3>',
            unsafe_allow_html=True
        )
        st.text(gramatura_data['text'])
        st.markdown("**Obraz fragmentu gramatury (od 'Przepisy/Składniki' do 'Pudełka'):**")

        if gramatura_data['image'] is not None:
            st.image(gramatura_data['image'], use_container_width=True)

            if gramatura_data['ingredients_data']:
                st.markdown("**Przeliczone gramatury składników:**")
                for result in gramatura_data['ingredients_data']:
                    if result['found_coefficient']:
                        st.markdown(f"**{result['name']}:**")
                        st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                        st.markdown(f"- Waga przed obróbką: {result['weight_before']:.0f} g (przy WpOT = {result['wpot_percent']}%)")
                    else:
                        st.markdown(f"**{result['name']}:**")
                        st.markdown(f"- Waga po obróbce (WpOT): {result['weight_wpot']:.0f} g")
                        st.markdown(f"- Nie znaleziono współczynnika WpOT dla tego składnika")
            else:
                st.warning("Nie udało się wyciągnąć składników z obrazu gramatury.")
        else:
            st.warning("Nie znaleziono fragmentu od 'Przepisy/Składniki' do 'Pudełka' dla tego ID w pliku gramatury.")
    else:
        st.warning(f"Nie znaleziono gramatury dla ID {selected_id}.")
