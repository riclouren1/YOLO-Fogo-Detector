%% Exportar modelo YOLOv2 para ONNX
% 1. Carregar o modelo treinado
load('detector_yolov2_incendios.mat');  % Certifica-te que já está treinado

% 2. Extrair a rede base da arquitetura do detetor
lgraph = layerGraph(detector.Network);

% 3. Converter para uma dlnetwork (necessária para exportação)
dlnet = dlnetwork(lgraph);

% 4. Exportar como modelo ONNX
exportONNXNetwork(dlnet, 'yolov2_modelo_exportado.onnx');

disp(" Modelo exportado com sucesso para yolov2_modelo_exportado.onnx");
