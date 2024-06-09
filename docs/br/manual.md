
## Teste Rápido

Marque "Enable" e selecione o Preset "example" na guia Config.

contrast: 1.2   
brightness: 0.9   
sharpeness: 1.5

Aplicar melhoria de borda   
Aplicar detalhamento de rosto   
Aplicar redimensionamento por pessoa   



## Opções padrão

Enabled (VERSION): Liga e desliga a funcionalidade.

### Substituir redimensionar e preencher

Quando realizar Img2Img, ao selecionar "Resize and fill", normalmente a imagem se expande 
para os lados ou para cima e para baixo, ou mantém a mesma proporção apenas alterando o tamanho.

Quando está ativado, a imagem sempre fica na parte inferior e se expande proporcionalmente 
para a esquerda, direita e para cima.

É eficaz aplicar quando não há margem na parte superior do rosto.   
Redimensionar excessivamente pode dificultar a obtenção de bons resultados.   
Recomenda-se usar em uma escala de aproximadamente 1.1 ou 1.2.   

<p>
<img src="https://i.ibb.co/j3WzZrc/00408-3188840002.png" width="40%">
<img src="https://i.ibb.co/ZWMWVFB/00409-3188840002.png" width="40%">
</p>

<br>
<br>
<br>

# Preprocessamento

Realiza um processo de pré-processamento antes de alterar a imagem.   
Dependendo das condições, pode interferir no processo hires.fix.

<a href="./preprocess.md">Preprocessamento</a>

# BMAB

Executa o detalhamento de pessoa, mão e rosto, ou realiza composição de imagem ou adição de ruído.

<a href="./bmab.md">bmab</a>

# Pós-processamento

Após o processamento da imagem, pode expandir o fundo ou fazer upscale dependendo do tamanho da pessoa.

<a href="./postprocess.md">Pós-processamento</a>


# API

Você pode usar a extensão BMAB ao chamar a funcionalidade API do stable diffusion webui.

<a href="./api.md">Guia de API</a>
