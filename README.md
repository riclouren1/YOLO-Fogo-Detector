-SafeForest - Sistema de Deteção de Incêndios com YOLOv2

Este projeto consiste no desenvolvimento de um sistema de deteção automática de incêndios (chama, fumo, chama_fumo, céu limpo e nuvem/neblina) com base em redes neurais convolucionais, utilizando o algoritmo YOLOv2 implementado no MATLAB.

---

-Objetivo

Desenvolver um classificador capaz de identificar precocemente sinais de incêndio florestal em imagens, com base em aprendizagem profunda (deep learning) e visão computacional.

---

-Tecnologias Utilizadas

- MATLAB + Deep Learning Toolbox
- YOLOv2 (You Only Look Once - versão 2)
- Image Labeler para anotação
- Exportação do modelo para ONNX

---

-Classes Detetadas

- chama
- fumo
- chama_fumo
- céu_limpo
- nuvem_neblina

---

-Resultados

- mAP@0.5: 0.720
- Tempo de inferência: < 1 segundo
- Testado com diversas imagens reais e simuladas
