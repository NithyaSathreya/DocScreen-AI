````
# AI-Based Fake Identity & Document Screening System

A research/prototype system for screening identity documents using multiple independent evidence sources:

- **PaddleOCR (PP-OCRv6)** — document text extraction [1][2]
- **PassportEye + Tesseract** — MRZ detection, OCR and validation [3][4]
- **InsightFace (Buffalo_L / SCRFD + ResNet50)** — face detection and face verification [5][6]
- **TruFor (SegFormer-B2 backbone)** — general-purpose image forgery detection and localization [7][8][9]
- **ADCD-Net** — document-specific forgery detection and localization using adaptive DCT features and hierarchical content disentanglement [10][11]
- **Evidence fusion** — combines the available evidence from both forensic methods and the other screening modules into a screening report

> **Purpose:** This project is intended for research and prototyping. Model outputs are screening evidence and should not be treated as definitive proof that a document is genuine or fraudulent.

## 1. Project Structure

```text
fake_identity_screening/
│
├── main.py
├── requirements.txt
├── README.md
│
├── input/
│   └── au_01.jpg
│
├── output/
│   ├── au_01.jpg.npz
│   ├── au_01.jpg_trufor_heatmap.jpg
│   └── au_01_report.json
│
├── modules/
│   ├── ocr.py
│   ├── mrz.py
│   └── face.py
│
├── networks/
│   └── paddleocr/
│       ├── PP-OCRv6_medium_det/
│       └── PP-OCRv6_medium_rec/
│
└── TruFor/
    └── TruFor_train_test/
        ├── test.py
        ├── weights/
        │   └── trufor.pth.tar
        ├── pretrained_models/
        │   └── segformers/
        │       └── mit_b2.pth
        └── lib/
            └── config/
                └── trufor_ph3.yaml
```

## 2. Create the Conda Environment

Create the dedicated Python environment:

```cmd
conda create -n docscreen python=3.10 -y
```

Activate it:

```cmd
conda activate docscreen
```

All following Python commands should be executed inside the `docscreen` environment.

## 3. Install Python Dependencies

Install the pinned dependencies:

```cmd
pip install -r requirements.txt
```

The requirements file contains the versions used for the working prototype, including PyTorch CUDA 11.8, InsightFace, ONNX Runtime GPU, PaddleOCR, PassportEye, Tesseract's Python wrapper, and TruFor dependencies.

### Important OpenCV note

The environment uses:

```text
opencv-contrib-python==4.10.0.84
```

Do not unnecessarily install another OpenCV distribution such as `opencv-python` or `opencv-python-headless` alongside it. Multiple OpenCV packages can overwrite the same `cv2` module and cause import/version problems.

### Important NumPy note

The current environment uses:

```text
numpy==1.26.4
```

Do not upgrade NumPy to 2.x without checking compatibility with the complete pipeline.

## 4. Install Tesseract OCR

Tesseract must be installed separately because it is a Windows executable and is not installed by `pip`.

### Windows documentation

Tesseract OCR Windows documentation:

urlTesseract OCR Windows documentationhttps://github.com/Strongcoffee2024/tesseract-OCR/blob/master/Home.md

### Download the installer

Use the Tesseract 5.5.0 Windows 64-bit installer:

urlDownload Tesseract 5.5.0 Windows installerhttps://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe

Install it using the Windows installer.

After installation, open an Anaconda Prompt and verify:

```cmd
tesseract --version
```

The working installation used in this project reports:

```text
tesseract v5.5.0.20241111
```

Then verify the Python wrapper:

```cmd
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

> **Important:** `pytesseract` in `requirements.txt` is only the Python wrapper. It does not install the Tesseract OCR executable.

If `tesseract` is not recognized, add the Tesseract installation directory to the Windows `PATH`.

## 5. Verify the Python Environment

### NumPy and OpenCV

```cmd
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

Expected working versions:

```text
OpenCV: 4.10.0
NumPy: 1.26.4
```

### InsightFace

```cmd
python -c "import insightface; print('InsightFace:', insightface.__version__)"
```

Expected:

```text
InsightFace: 1.0.1
```

### PyTorch and GPU

```cmd
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The development system uses:

```text
torch==2.5.1+cu118
torchvision==0.20.1+cu118
torchaudio==2.5.1+cu118
```

### ONNX Runtime

```cmd
python -c "import onnxruntime as ort; print('ORT:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

The development environment uses:

```text
onnxruntime-gpu==1.21.0
```

### PassportEye

```cmd
python -c "import passporteye; print('PassportEye: OK')"
```

### TruFor dependencies

```cmd
python -c "import timm, yacs; print('TruFor dependencies: OK')"
```

## 6. Models

### PaddleOCR

The PaddleOCR models used by the project are stored locally under:

```text
networks/paddleocr/
```

including:

```text
PP-OCRv6_medium_det/
PP-OCRv6_medium_rec/
```

The model directories should be available before running the OCR module. The PP-OCRv6 detection and recognition models are the OCR networks used by this project.[1][2]

### InsightFace

The Buffalo_L model is stored/downloaded under the user's InsightFace model directory:

```text
~/.insightface/models/buffalo_l/
```

The model package includes the face detection, landmark, gender/age, and face-recognition ONNX models used by InsightFace. The Buffalo_L model package uses SCRFD for face detection and ResNet50-based face recognition.[5][6]

### TruFor

The main TruFor checkpoint is:

```text
TruFor/TruFor_train_test/weights/trufor.pth.tar
```

The SegFormer backbone is:

```text
TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth
```

The TruFor inference code is located at:

```text
TruFor/TruFor_train_test/test.py
```

### ADCD-Net

The project also uses **ADCD-Net** as a document-specific image forgery detection and localization method. ADCD-Net combines RGB and DCT forensic traces with hierarchical content disentanglement and an adaptive mechanism designed to improve robustness to document-image distortions such as resizing and cropping.[10][11]

The project uses the ADCD-Net checkpoint:

```text
ADCDNet.pth
```

and the associated DocRes backbone checkpoint:

```text
docres.pkl
```

The current project implementation is integrated through:

```text
modules/adcd.py
```

ADCD-Net is used independently of TruFor. The two methods provide separate forensic evidence and should not be treated as interchangeable models.

## 7. Run the Complete System

**The normal way to run the complete fake identity and document screening system is through `main.py`.**

From the project root:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

This single command runs the complete pipeline:

```text
                         Input Document
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          PaddleOCR        PassportEye       InsightFace
          Text OCR             MRZ             Face Detection
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                           TruFor
                    Tampering Analysis
                               │
                               ▼
                         Evidence Fusion
                               │
                               ▼
                          JSON Report
```

### Command Arguments

| Argument | Description | Example |
|---|---|---|
| `--input` | Input identity/passport document image | `input/au_01.jpg` |
| `--lang` | OCR language | `en` |
| `--gpu` | GPU device index; use `-1` for CPU | `0` |
| `--output` | Directory for generated results | `output` |

### NVIDIA GPU

For GPU inference:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

### CPU

To force CPU inference:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu -1 --output output
```

### Normal Workflow

After the environment and models have been installed, the normal workflow is simply:

```cmd
conda activate docscreen
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

The individual scripts are intended for development, testing, and troubleshooting.

## 8. Pipeline Components

### 8.1 PaddleOCR

PaddleOCR extracts visible document text using the PP-OCRv6 detection and recognition models.[1][2]

Typical information includes:

```text
document text
names
dates
document numbers
nationality
address/details
```

OCR output can subsequently be compared against MRZ information.

### 8.2 PassportEye + Tesseract

PassportEye is used for MRZ detection and parsing, while Tesseract provides OCR support.[3][4]

Typical MRZ output includes:

```text
document_type
country
passport_number
nationality
date_of_birth
expiration_date
sex
surname
given_names
personal_number
valid_number
valid_date_of_birth
valid_expiration_date
valid_personal_number
valid_composite
valid_score
raw_text
```

MRZ check digits provide useful consistency evidence, but valid MRZ data alone does not prove that a passport is genuine.

### 8.3 InsightFace

InsightFace is used to detect faces in the document image and to perform face verification when a legitimate reference image is available.[5][6]

The prototype reports:

```text
number of detected faces
bounding boxes
detection confidence
```

Face detection is not the same as identity verification. A future version can compare the document portrait against an authorized reference image when such a reference is legitimately available.

### 8.4 TruFor

TruFor provides image-forensics evidence for possible image manipulation and localization.[7][8]

The model produces:

```text
map
imgsize
score
conf
```

Meaning:

```text
score  → overall TruFor detection score
map    → pixel-level manipulation prediction
conf   → confidence map
```

For the development test document, TruFor produced:

```text
Score: 0.9534304141998291
Map mean: 0.11072701
Confidence mean: 0.9575034
Image size: 1521 x 2167
```

The corresponding heatmap showed strong responses around the main portrait, the smaller portrait, portions of the MRZ, and some security-pattern/background regions.

These values must not be interpreted as a simple:

```text
score > threshold = fake
```

rule. TruFor is a forensic evidence source and should be interpreted together with the other document evidence.

## 9. ADCD-Net

ADCD-Net is the second image-forensics method used by the project. Unlike TruFor, which is a general-purpose image forgery detection and localization method, ADCD-Net is specifically designed for **document image forgery localization**.[10][11]

ADCD-Net adaptively uses RGB and DCT forensic traces and incorporates hierarchical content disentanglement to address characteristics of document images, including structured text and relatively uniform document backgrounds. The method is also designed to improve robustness to distortions such as resizing and cropping.[10]

The project uses ADCD-Net independently from TruFor:

```text
Input document
      │
      ├──────────────► TruFor
      │                 │
      │                 └── Forgery evidence
      │
      └──────────────► ADCD-Net
                        │
                        └── Document-forgery evidence
```

The ADCD-Net implementation produces forensic evidence including a document-level score, pixel-level tampering information, and an alignment-related score. These outputs are interpreted together with the OCR, MRZ, face, and other available evidence.

> **Important:** TruFor and ADCD-Net are two independent forensic methods. A high score from either method is evidence of a possible anomaly, not definitive proof of fraud. The final screening decision should consider multiple evidence sources.

## 9. TruFor Output

A successful TruFor run produces an `.npz` result containing:

```text
map
imgsize
score
conf
```

Example:

```text
output/
└── au_01.jpg.npz
```

The development result contained:

```text
map      (1521, 2167)
conf     (1521, 2167)
score    scalar
imgsize  (2,)
```

The `map` and `conf` arrays preserve the spatial dimensions of the processed document.

## 10. TruFor Visualization

The separate visualization utility can convert the TruFor manipulation map into a heatmap.

Example:

```cmd
python visualize_trufor.py --input input/au_01.jpg --result output/au_01.jpg.npz --output output/au_01_trufor_heatmap.jpg
```

Output:

```text
output/au_01_trufor_heatmap.jpg
```

The heatmap helps identify where TruFor is responding strongly instead of relying only on the global score.

## 11. Output Report

A successful complete run should produce results similar to:

```text
output/
├── au_01.jpg.npz
├── au_01.jpg_trufor_heatmap.jpg
└── au_01_report.json
```

The JSON report combines the evidence returned by:

```text
OCR
MRZ validation
Face detection
TruFor
ADCD-Net
```

Example report structure:

```json
{
    "input": "input/au_01.jpg",
    "ocr": {},
    "mrz": {},
    "face": {},
    "trufor": {}
}
```

The exact contents depend on which modules successfully process the input.

## 12. Evidence Fusion

The final screening decision should not be based on a single model.

The intended architecture is:

```text
                         DOCUMENT
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
            OCR             MRZ           FACE
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                          TruFor
                            │
                            ▼
                     Evidence Fusion
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
               PASS       REVIEW     SUSPICIOUS
```

Potential future checks include:

- OCR ↔ MRZ field consistency
- MRZ check-digit validity
- Portrait location and quality
- Number and location of detected faces
- TruFor overall score
- TruFor suspicious-area percentage
- TruFor region-level scores
- ADCD-Net document-level score
- ADCD-Net pixel-level localization
- ADCD-Net alignment score
- Cross-method agreement between TruFor and ADCD-Net
- Document layout consistency
- Face comparison when an authorized reference image exists

The system should distinguish between **evidence of anomaly** and a definitive claim of fraud.

## 13. Current Working Environment

The prototype was developed and tested with:

```text
Python                 3.10
NumPy                  1.26.4
OpenCV                 4.10.0
PyTorch                2.5.1+cu118
TorchVision             0.20.1+cu118
TorchAudio              2.5.1+cu118
InsightFace             1.0.1
ONNX Runtime GPU        1.21.0
PaddleOCR               3.7.0
PaddlePaddle            3.3.1
PaddleX                 3.7.2
PassportEye             2.2.2
Tesseract               5.5.0.20241111
scikit-learn             1.7.2
scikit-image             0.25.2
timm                    1.0.20
yacs                    0.1.8
```

Minimum Requirement of GPU : The NVIDIA development system with GTX 1050 Ti with 4 GB VRAM.

TruFor successfully ran GPU inference on this hardware, although inference on large document images can take several minutes and uses most of the available VRAM.

## 14. Compatibility Notes

Do not casually upgrade the core packages in the working environment.

In particular, avoid changing these without testing the complete pipeline:

```text
numpy
opencv-contrib-python
torch
torchvision
onnxruntime-gpu
paddlepaddle
paddleocr
paddlex
```

The environment required careful compatibility handling.

### ONNX Runtime

The environment contains:

```text
onnxruntime-gpu==1.21.0
```

The available providers may include:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

However, provider availability does not necessarily mean that CUDA execution can initialize successfully. CUDA/cuDNN DLL compatibility must be verified separately.

### TruFor and ADCD-Net

TruFor and ADCD-Net are independent forensic methods in this project. TruFor uses its own PyTorch inference pipeline, while ADCD-Net is integrated through `modules/adcd.py`.

An issue in one forensic method should not be used as a reason to modify the other method. The two methods should be tested independently before changing the complete screening pipeline.

### TruFor and PyTorch CUDA

TruFor uses PyTorch CUDA directly and is independent of the ONNX Runtime CUDA provider used by InsightFace.

Therefore an ONNX Runtime CUDA initialization problem does not necessarily prevent TruFor from using the NVIDIA GPU.

## 15. Troubleshooting

### `tesseract` is not recognized

Verify:

```cmd
tesseract --version
```

If Windows cannot find it, add the Tesseract installation directory to `PATH`.

### `ModuleNotFoundError: No module named 'passporteye'`

Install:

```cmd
pip install passporteye
```

Then verify:

```cmd
python -c "import passporteye; print('PassportEye: OK')"
```

### `ModuleNotFoundError: No module named 'yacs'`

Install:

```cmd
pip install yacs --no-deps
```

### `ModuleNotFoundError: No module named 'timm'`

Install:

```cmd
pip install timm --no-deps
```

### TruFor cannot find its checkpoint

Verify:

```cmd
dir TruFor\TruFor_train_test\weights
```

The directory should contain:

```text
trufor.pth.tar
```

### TruFor creates an output path such as `C:\au_01.jpg.npz`

This is a Windows path-handling issue in the original TruFor test script when a single absolute input file is supplied.

The project version of `test.py` should use the corrected output-path handling so that:

```text
input\au_01.jpg
```

produces:

```text
output\au_01.jpg.npz
```

### TruFor appears to hang

Check GPU usage:

```cmd
nvidia-smi
```

If Python is using high GPU utilization and most of the GPU memory, TruFor is processing the image.

Large images can take significant time on GPUs with 4 GB VRAM.

## 16. Individual Component Testing

The complete system should normally be run through:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

Individual scripts can be used when debugging a specific component.

For example, the official TruFor test program can be run from:

```text
TruFor\TruFor_train_test
```

using:

```cmd
python test.py -g 0 -in "C:\path\to\input\au_01.jpg" -out "C:\path\to\output" TEST.MODEL_FILE weights/trufor.pth.tar
```

This is primarily for TruFor component testing rather than the normal application workflow.

## 16. Individual Module Testing

Each pipeline component can be tested independently before running the complete system. This is recommended when setting up the environment or troubleshooting a particular component.

### 16.1 Test PaddleOCR

From the project root:

```cmd
python test_ocr.py --input input/au_01.jpg --lang en
```

This tests:

```text
Input image
    ↓
PaddleOCR
    ↓
Extracted document text
```

Use this test when troubleshooting OCR installation, model loading, or text extraction.

### 16.2 Test MRZ Extraction and Validation

```cmd
python test_mrz.py --input input/au_01.jpg
```

This tests:

```text
Input passport
    ↓
PassportEye + Tesseract
    ↓
MRZ detection
    ↓
MRZ parsing
    ↓
Check-digit validation
```

Typical output includes:

```text
passport_number
date_of_birth
expiration_date
nationality
surname
given_names
valid_number
valid_date_of_birth
valid_expiration_date
valid_personal_number
valid_composite
valid_score
raw_text
```

### 16.3 Test Face Detection

```cmd
python test_face.py --input input/au_01.jpg
```

This tests:

```text
Input image
    ↓
InsightFace
    ↓
Face detection
    ↓
Bounding boxes + confidence
```

Example:

```text
Faces detected: 3
Face 1: bbox=[...], score=...
Face 2: bbox=[...], score=...
Face 3: bbox=[...], score=...
```

### 16.4 Test Face Verification Against a Registered Face

Face detection and face verification are separate operations.

`test_face.py` answers:

```text
Is a face present in the document?
```

`test_verification.py` answers:

```text
Does the face in the document match the registered/reference face?
```

The registered face should be provided as a separate reference image.

Example:

```cmd
python test_verification.py --input input/au_01.jpg --reference registered/au_01.jpg
```

The verification pipeline is:

```text
Document image
      ↓
Face detection
      ↓
Document portrait
      ↓
Face embedding
      ↓
Compare with registered face
      ↑
Registered/reference image
      ↓
Similarity / verification result
```

This is an important part of identity screening because a document may contain a valid-looking face while the face does not belong to the registered person.

A typical project structure is:

```text
fake_identity_screening/
│
├── input/
│   └── au_01.jpg
│
├── registered/
│   └── au_01.jpg
│
└── test_verification.py
```

> The exact command-line arguments should match the implementation of the project's `test_verification.py`. The example above assumes the script uses `--input` for the document image and `--reference` for the registered face.

### 16.5 Test TruFor

TruFor can be tested independently from its test directory.

Change to:

```cmd
cd TruFor\TruFor_train_test
```

Then run:

```cmd
python test.py -g 0 -in "Path to\fake_identity_screening\input\au_01.jpg" -out "Path To\fake_identity_screening\output" TEST.MODEL_FILE weights/trufor.pth.tar
```

This produces:

```text
output\au_01.jpg.npz
```

The `.npz` result contains:

```text
map
imgsize
score
conf
```

### 16.5 Visualize TruFor

Return to the project root and run:

```cmd
python visualize_trufor.py --input input/au_01.jpg --result output/au_01.jpg.npz --output output/au_01_trufor_heatmap.jpg
```

The resulting heatmap shows the regions where TruFor produces stronger manipulation responses.

### 16.6 Test ADCD-Net

ADCD-Net can be tested independently from the other screening components.

The integrated project implementation is:

```text
modules/adcd.py
```

The required model files are:

```text
ADCDNet.pth
docres.pkl
```

The ADCD-Net test should verify:

```text
Input document
    ↓
ADCD-Net
    ↓
Forgery score
    ↓
Pixel-level localization
    ↓
Alignment evidence
```

The exact command should match the project's ADCD-Net test script if one is provided. The purpose of this test is to verify ADCD-Net independently before running the complete screening pipeline.

### 16.6 Complete System Test

After the individual components have been verified, run the complete pipeline:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

The recommended testing order is:

```text
test_ocr.py
    ↓
test_mrz.py
    ↓
test_face.py
    ↓
test_verification.py
    ↓
TruFor test.py
    ↓
visualize_trufor.py
    ↓
ADCD-Net test
    ↓
main.py
```

Testing the modules individually makes it easier to identify whether a problem is related to OCR, MRZ processing, face detection, TruFor, or the final integration.

## 17. Current Project Status

The following components have been successfully tested during development:

```text
Conda environment        OK
Tesseract                OK
PassportEye              OK
MRZ extraction           OK
MRZ check validation     OK
InsightFace              OK
Face detection           OK
TruFor checkpoint        OK
TruFor model loading     OK
TruFor GPU inference     OK
TruFor localization map  OK
TruFor heatmap            OK
ADCD-Net model            OK
ADCD-Net GPU inference    OK
```

The main remaining development work is the **evidence-fusion/decision layer**, including region-level TruFor analysis and consistency checks between OCR, MRZ, face information, and image-forensics results.

## 18. References

[1] PaddlePaddle, **PaddleOCR: Open-Source OCR Toolkit**. GitHub.  
https://github.com/PaddlePaddle/PaddleOCR

[2] PaddlePaddle, **PP-OCRv6**. PaddleOCR Documentation.  
https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html

[3] Tretyakov, K., **PassportEye: Extraction of machine-readable zone information from passports, visas and ID cards via OCR**. GitHub.  
https://github.com/konstantint/PassportEye

[4] Smith, R. (2007), **An Overview of the Tesseract OCR Engine**. Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR).  
https://github.com/tesseract-ocr/tesseract

[5] InsightFace, **InsightFace: 2D and 3D Face Analysis Project**. GitHub.  
https://github.com/deepinsight/insightface

[6] InsightFace, **Model Zoo – Buffalo_L**. GitHub.  
https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md

[7] Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., & Verdoliva, L. (2023), **TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization**. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20606–20615.  
https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.html

[8] GRIP-University of Naples Federico II, **TruFor: Official PyTorch Implementation**. GitHub.  
https://github.com/grip-unina/TruFor

[9] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021), **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers**. arXiv:2105.15203.  
https://arxiv.org/abs/2105.15203

[10] Wong, K., Zhou, J., Wu, H., Si, Y.-W., & Zhou, J. (2025), **ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement**. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 19280–19289.  
https://openaccess.thecvf.com/content/ICCV2025/html/Wong_ADCD-Net_Robust_Document_Image_Forgery_Localization_via_Adaptive_DCT_Feature_ICCV_2025_paper.html

[11] KahimWong, **ADCD-Net: Official Implementation**. GitHub.  
https://github.com/KAHIMWONG/ADCD-Net

## 19. Licensing and Usage

### TruFor

TruFor is distributed by the Image Processing Research Group of the University Federico II of Naples under its stated license terms.

Review the included files before using TruFor:

```text
TruFor/TruFor_train_test/LICENSE.txt
TruFor/TruFor_train_test/LICENSE_CMX.txt
```

The TruFor package specifies nonprofit use.

### ADCD-Net

ADCD-Net is released under the MIT License in the official repository. Review the official repository and included license before redistribution or commercial deployment.

### Other components

PaddleOCR, PaddlePaddle, InsightFace, PassportEye, Tesseract, PyTorch, ADCD-Net, and the other dependencies have their own licenses. Review their respective license terms before redistribution or commercial deployment.

## 20. Quick Start

For a fresh Windows setup:

```cmd
conda create -n docscreen python=3.10 -y
conda activate docscreen
pip install -r requirements.txt
```

Install Tesseract using the Windows installer described in Section 4.

Verify:

```cmd
tesseract --version
```

Then place the input document in:

```text
input\au_01.jpg
```

Finally run the complete system:

```cmd
python main.py --input input/au_01.jpg --lang en --gpu 0 --output output
```

Results will be written to:

```text
output\
```

The main report is:

```text
output\au_01_report.json
```

````