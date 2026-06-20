# Lensless Imaging Project

Смысл проекта в восстановлении изображения с безлинзовой камеры, по размазанному снимку с сенсора и известному PSF, с помощью ADMM и его улучшенных обучаемых версий.

Реализованы методы:

- `ADMM100`: необучаемый ADMM на 100 итераций по статье.
- `UnrolledADMM20`: ADMM на 20 итераций с обучаемыми параметрами.
- `LeADMM5Pre`: DRUNet preprocessing перед 5 итерациями Le-ADMM.
- `LeADMM5Post`: DRUNet postprocessing после 5 итераций Le-ADMM.
- `LeADMM5PrePost`:  preprocessing + 5 итераций Le-ADMM + postprocessing (тоже DRUNet).


Проект написан по статьям:
[Lensless Imaging Reconstruction](https://arxiv.org/pdf/2502.01102) и [Le-ADMM](https://arxiv.org/pdf/1908.11502)
- Веса модели: [artii-ml/lensless-imaging](https://huggingface.co/artii-ml/lensless-imaging)
- Отчёт: [comet-ml](https://www.comet.com/artii/final-project/reports/ox65dPEN9sgIT96buXlqG86NO)

## Результаты

Метрики моделей после 20 эпох обучения. (кроме необучаемого ADMM-100)

| Model |  PSNR ↑ | SSIM ↑ | MSE ↓ | LPIPS ↓ | Inference time, sec/image ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ADMM100` | 9.21   | 0.0410 |0.1344 | 0.772 | 1.112 |
| `UnrolledADMM20` | 13.04 | 0.3299 | 0.0518 | 0.737 | 0.226 |
| `LeADMM5Pre` | 14.02 | 0.2676 | 0.0410 | 0.679 | **0.132** |
| `LeADMM5Post` | 16.26 | 0.4379 | 0.0252 |0.570| 0.134 |
| `LeADMM5PrePost` | **16.78** | **0.4626**| **0.0227** | **0.551** | 0.170 |

Лучшая модель выбивает оценку 10 по метрике PSNR.

## Demo 
Демо - ноутбук, готовый к запуску в Google colab. 

Может принять url путь к zip файлу датасета пользователя на  Google Drive. Датасет должен быть определённой структуры, описанной ниже в Данных. 

Берёт несколько размытых изображений из заданного датасета, строит реконструкции по размытому изображению и маске. выводит полученные изображения на экран и считает метрики, если дан оригинал изображения.

Базово задан путь к zip файлу test части DigiCam датасета, то есть при параметре count >=1500 запустится полный инференс выбранной модели на тестовом датасете, и будут воспроизведены итоговые результаты. 


## Структура

```text
lensless-imaging
├── train.py  # скрипт для обучения
├── inference.py  # скрипт для инференса моделей
├── calculate_metrics.py  # вспомогательный скрипт для подсчёта метрик
├── demo.ipynb  # demo с демонстрацией работы
├── requirements.txt  # зависимости
├── src/
|   ├── configs  # конфиги параметров 
|   ├── datasets  #классы датасетов
|   ├── loss  # функции лосса
|   ├── metrics  # метрики
|   ├── model  # все используемые модели
|   ├── scripts  # вспомогательные скрипты
|   ├── trainer  # класс тренера для обучения
|   └── utils  #  отдельные полезные функции
└── lensless_helpers/  # вспомогательные функции

```

## Загрузка и настройка

```bash
git clone https://github.com/cherrypy1/lensless-imaging lensless_project
cd lensless_project
```

Установка необходимых зависимостей:

```bash
pip install -r requirements.txt
```

## Данные

Для обучения и оценки на [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K):

```bash
python src/scripts/download_digicam.py --output data/digicam
python src/scripts/download_masks.py --output data/masks
```

Пользовательская папка с данными для demo должна иметь структуру :

```text
PathToDataDir
├── lensless
│   ├── ImageID1.png
│   ├── ImageID2.png
│   .
│   .
│   .
│   └── ImageIDn.png
├── masks
│   ├── ImageID1.npy
│   ├── ImageID2.npy
│   .
│   .
│   .
│   └── ImageIDn.npy
└── lensed # оригинальные изображения, может не существовать
    ├── ImageID1.png
    ├── ImageID2.png
    .
    .
    .
    └── ImageIDn.png
```


## Веса моделей

Скачать все checkpoint-файлы:

```bash
python src/scripts/download_checkpoint.py --name all
```

Скачать только одну модель:

```bash
python src/scripts/download_checkpoint.py --name leadmm5_pre_post
```
Доступные имена: 

`unrolled_admm20` , `leadmm5_pre`, `leadmm5_post`, `leadmm5_pre_post`

Файлы сохраняются в `checkpoints/`.

## Обучение

Стоит указать Comet API key для логгирования:

```bash
export COMET_API_KEY="YOUR_COMET_API_KEY"
```

Обучение моделей:

```bash
python train.py -cn=unrolled_admm20
python train.py -cn=leadmm5_pre
python train.py -cn=leadmm5_post
python train.py -cn=leadmm5_pre_post
```

Workspace Comet можно задать через Hydra:

```bash
python train.py -cn=leadmm5_pre_post writer.workspace=YOUR_WORKSPACE
```

Checkpoint сохраняются в:

```text
saved/{run_name}/checkpoint-epoch{num_epoch}.pth
```

В базовой настройке последний чекпоинт сохраняется после 20 эпохи обучения

## Инференс и подсчёт метрик

Команды написаны и описаны в demo. Можно считать demo основным источником запуска инференса. 

Можно взять команды оттуда и запускать их другим способом.