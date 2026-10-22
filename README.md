# PaddyCare — Multimodal Starter Project

PaddyCare is a student prototype that combines:
- a Streamlit user interface;
- Lex/Flex lexical analysis (`symptom.l`);
- Yacc/Bison grammar parsing (`symptom.y`);
- a future image classifier for paddy leaf images;
- a future NLP model for free-form symptom descriptions.

## Important status note

The web interface runs before the AI model is ready. If `symptom_parser.exe` has not been compiled, the interface uses a small Python fallback to demonstrate symptom-term extraction. Once compiled, it invokes the actual Lex/Yacc executable. The image upload currently displays the image but does not claim a disease prediction. A trained and evaluated model must be added later.

## 1. Create a virtual environment (Windows PowerShell)

Open the project folder in VS Code, then open Terminal > New Terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 2. Compile and run Lex/Yacc on Windows

Recommended path: install **MSYS2** from https://www.msys2.org/ and open the **MSYS2 UCRT64** terminal. Install the compiler and parser tools:

```bash
pacman -Syu
```

If the terminal asks you to close and reopen, do so, then run:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc flex bison
```

Navigate to the PaddyCare project directory from the UCRT64 terminal. Windows drives are mounted under `/c`, `/d`, etc. For example:

```bash
cd /c/Users/YOUR_WINDOWS_USERNAME/path/to/PaddyCare_Starter
```

Generate the parser and scanner, then compile:

```bash
bison -d -o symptom.tab.c symptom.y
flex -o lex.yy.c symptom.l
gcc -o symptom_parser.exe symptom.tab.c lex.yy.c
```

The `.l` file is the Lex/Flex specification; the `.y` file is the Yacc/Bison grammar. The executable is called by the Streamlit app.

Test the parser:

```bash
echo "Brown spots are spreading and leaves are drying" | ./symptom_parser.exe
```

Expected output includes:

```text
brown_spots=1
yellowing=0
drying=1
blight=0
rust=0
wilting=0
lesions=0
mildew=0
```

Copy the generated `symptom_parser.exe` into the project root (the same folder as `app.py`) if you compiled it from another directory. Restart Streamlit after compiling.

## 3. Dataset and AI model

No dataset is bundled with this starter. Before training, choose a reputable paddy/rice leaf disease dataset, check its license and class labels, and split it into training, validation, and test sets. Then we will implement the image training pipeline and evaluate performance on held-out images.

## 4. Planned architecture

1. Image input → image preprocessing → trained computer-vision encoder.
2. Symptom text → Lex tokenization → Yacc grammar parsing for supported symptom patterns.
3. Free-form text → NLP encoder (to be added and evaluated).
4. Image and text features → fusion / decision logic.
5. Prediction + confidence + evidence-based management guidance.

Lex/Yacc parse the supported symptom patterns; they do not themselves perform image recognition or full natural-language understanding. Disease predictions and treatment advice must be validated before real-world use.
