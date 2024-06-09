
# Pré-processamento (Preprocess)

## Contexto

Especifique o Checkpoint e o VAE a serem usados no BMAB.
Algumas funcionalidades podem definir seu próprio Checkpoint e VAE.
Uma vez alterado, o Checkpoint continuará a ser usado em processos subsequentes.

<img src="https://i.ibb.co/VTP5ddx/2023-11-12-3-48-54.png">

#### Multiplicador de ruído txt2img para hires.fix (txt2img noise multiplier for hires.fix)

Pode adicionar ruído na fase hires.fix.

#### Multiplicador de ruído extra txt2img para hires.fix (EXPERIMENTAL) (txt2img extra noise multiplier for hires.fix (EXPERIMENTAL))

Pode adicionar ruído extra na fase hires.fix.

#### Filtro hires.fix antes do upscaler (Hires.fix filter before upscaler)

Pode aplicar um filtro antes do upscaler na fase hires.fix.

#### Filtro hires.fix depois do upscaler (Hires.fix filter after upscaler)

Pode aplicar um filtro depois do upscaler na fase hires.fix.


## Resample (EXPERIMENTAL)

Função de auto-resampling. Gera a imagem txt2img -> hres.fix e realiza o processo novamente txt2img -> hires.fix com
ControlNet Tile Resample. Pode ser utilizado nas seguintes situações:

* Quando há grande diferença nos resultados entre dois modelos.
* Quando as proporções das pessoas diferem entre dois modelos.
* Quando as versões dos modelos são diferentes (SDXL, SD15).

<table>
<tr>
<td>txt2img->hires.fix</td>
<td>Resample + BMAB Basic</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/VxPfgN0/00153-3939130001-before-resample.png"></td>
<td><img src="https://i.ibb.co/XZ9gHHN/00154-3939130001.png"></td>
</tr>
</table>

<img src="https://i.ibb.co/5hWtbmZ/e822842f656d73757ee65713317f7ba9d947472d3fe94fc3ceffc72aee31064d.jpg">
BMAB resample image by [padapari](https://www.instagram.com/_padapari_/)

<br>
<br>
<br>
<br>


<img src="https://i.ibb.co/9hD81hd/resample.png">

#### Ativar auto-resampling (EXPERIMENTAL) (Enable auto-resampling (EXPERIMENTAL)

Pode ativar e desativar esta função.

#### Salvar imagem antes do processamento (Save image before processing)

Quando a imagem gerada por txt2img -> hires.fix é inserida no BMAB para pós-processamento, 
salva a imagem antes de processá-la. O sufixo "-before-resample" será adicionado à imagem.

#### Checkpoint

Pode especificar o Checkpoint SD. Se não especificado, usa o Checkpoint definido anteriormente.
Não retorna ao original após a conclusão do processo.

#### SD VAE

Pode especificar o VAE SD. Se não especificado, usa o VAE definido anteriormente.
Não retorna ao original após a conclusão do processo.

#### Método de Resample (Resample method)

É possível escolher o método de Resample.

txt2img-1pass: Executa txt2img sem hires.fix.
txt2img-2pass: Executa txt2img com hires.fix. Por padrão, o hires.fix deve ser executado ao exportar a imagem.
img2img-1pass: Executa img2img.

#### Filtro de Resample (Resample filter)

Após completar o Resample, pode chamar um código de filtro externo para realizar transformações adicionais na imagem.


#### Prompt de Resample (Resample prompt)

Prompt a ser usado durante o processo de resampling. Se vazio, é igual ao prompt principal. 
Use "#!org!#" para substituir o prompt principal. Pode adicionar mais ao prompt após "#!org!#".
ex) #!org!#, soft light, some more keyword

#### Prompt negativo de Resample (Resample negative prompt)

Prompt negativo a ser usado durante o processo de resampling. Se vazio, é igual ao prompt negativo principal.

#### Método de Sampling (Sampling method)

Especifique o método de sampling a ser usado no processo. Se não especificado, usa o mesmo sampler do processo anterior.

#### Upscaler

O upscaler a ser usado se estiver utilizando hires.fix.

#### Etapas de sampling de Resample (Resample sampling steps)

Especifique as etapas de sampling a serem usadas no processo de resample.
(Recomendado 20)

#### Escala CFG de Resample (Resample CFG scale)

Especifique o valor da escala CFG a ser usado no processo de resample.
Não suporta threshold dinâmico.

#### Força de denoising de Resample (Resample denoising strength)

Especifique a força de denoising a ser usada no processo de resample.
(Recomendado 0.4)

#### Força de Resample (Resample strength)

Valores próximos a 0 se afastam da imagem de entrada, enquanto valores próximos a 1 se assemelham mais à imagem original.

#### Início de Resample (Resample begin)

Ponto de início para aplicação durante a etapa de sampling.

#### Fim de Resample (Resample end)

Ponto de término para aplicação durante a etapa de sampling.








## Pretraining (EXPERIMENTAL)

Detalhamento de pretraining. Aplica um modelo de detecção pretreinado com ultralytics e utiliza isso 
para aplicar prompt e negative prompt, permitindo desenhar partes da imagem com mais detalhes.

<img src="https://i.ibb.co/Qkx6rQK/pretraining.png"/>

#### Ativar detalhamento de pretraining (EXPERIMENTAL) (Enable pretraining detailer)

Pode ativar e desativar esta função.

#### Ativar pretraining antes do hires.fix (Enable pretraining before hires.fix)

Realiza o detalhamento de pretraining antes do hires.fix.

#### Modelo de Pretraining (Pretraining model)

Pode especificar o modelo de detecção treinado com ultralytics (*.pt).
O arquivo deve estar em stable-diffusion-webui/models/BMAB para aparecer na lista.


#### Prompt de Pretraining (Pretraining prompt)

Prompt a ser usado durante o processo de detalhamento de pretraining. Se vazio, é igual ao prompt principal. 
Use "#!org!#" para substituir o prompt principal. Pode adicionar mais ao prompt após "#!org!#".
ex) #!org!#, soft light, some more keyword

#### Prompt negativo de Pretraining (Pretraining negative prompt)

Prompt negativo a ser usado durante o processo de detalhamento de pretraining. Se vazio, é igual ao prompt negativo principal.

#### Método de Sampling (Sampling method)

Especifique o método de sampling a ser usado no processo. Se não especificado, usa o mesmo sampler do processo anterior.


#### Etapas de sampling de Pretraining (Pretraining sampling steps)

Especifique as etapas de sampling a serem usadas no processo de pretraining.
(Recomendado 20)

#### Escala CFG de Pretraining (Pretraining CFG scale)

Especifique o valor da escala CFG a ser usado no processo de pretraining.
Não suporta threshold dinâmico.

#### Força de denoising de Pretraining (Pretraining denoising strength)

Especifique a força de denoising a ser usada no processo de pretraining.
(Recomendado 0.4)

#### Dilatação de Pretraining (Pretraining dilation)

Aumenta o tamanho do retângulo de detecção em um valor especificado.

#### Pretraining box threshold

Decide o valor de detecção do detector. Valores abaixo do padrão 0.35 serão excluídos como não sendo face.
É o valor de confiança da predição do ultralytics.



## Edge enhancement

Funcionalidade para aumentar a nitidez ou os detalhes das bordas da imagem.

**<span style="color: red">Não funciona se o upscaler for da série Latent. (Recomenda-se R-ESRGAN, 4x-UltraSharp)</span>**

<img src="https://i.ibb.co/4sjB1Lr/edge.png">

Configurações recomendadas

* Limite baixo de borda: 50
* Limite alto de borda: 200
* Força de borda: 0.5

<p>
<img src="https://i.ibb.co/Wsw2Wrh/00598-1745587019.png" width="40%">
<img src="https://i.ibb.co/z4nCW9Z/00600-1745587019.png" width="40%">
</p>

Enabled: CHECK!!   

Contrast : 1.2   
Brightness : 0.9   
Sharpeness : 1.5   

Enable edge enhancement : CHECK!!   
Edge low threshold : 50   
Edge high threshold : 200   
Edge strength : 0.5   




## Resize

Funciona durante o processo intermediário de txt2img -> hires.fix.
Se usado em img2img, funciona antes do início do processo.

Ajusta a proporção entre a altura da pessoa mais alta e a altura da imagem, se a proporção exceder o valor configurado.
Se o valor configurado for 0.90 e a proporção da pessoa for 0.95, ajusta a proporção para 0.90, expandindo o fundo.
O fundo é expandido conforme especificado no Alignment.

Ajusta a imagem antes da fase hires.fix durante a execução de txt2img.
Isso é para garantir que a imagem modificada seja suavemente transformada na fase hires.fix.
**<span style="color: red">Use uma força de denoising entre 0.6 e 0.7 para evitar distorções na imagem ao redor.</span>**
**<span style="color: red">Não funciona se o upscaler for da série Latent. (Recomenda-se R-ESRGAN, 4x-UltraSharp)</span>**

#### Método (Method)

Pode especificar o método de Resize.

* Stretching: Expande o fundo simplesmente esticando as bordas da imagem.
* Inpaint: Usa uma máscara para realizar inpainting apenas nas áreas esticadas.
* Inpaint+lama: Usa o modelo inpaint+lama do Controlnet para redesenhar as áreas expandidas.
* Inpaint_only: Usa apenas o modelo inpaint_only do Controlnet para redesenhar as áreas expandidas.


#### Alinhamento (Alignment)

Decide em que direção alinhar a imagem original ao expandir.

<img src="https://i.ibb.co/g62KhZQ/align.png">

#### Filtro de Resize (Resize filter)

Após completar o Resize, pode chamar um código de filtro externo para realizar transformações adicionais na imagem.


#### Proporção intermediária de Resize por pessoa (esize by person intermediate)

A proporção do tamanho da pessoa. Se exceder este valor, o fundo será expandido para ajustar à proporção especificada.



<table>
<tr>
<td>Original</td>
<td>Resize 0.7</td>
<td>Resize 0.5</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/XttbBz0/00133-3615254454.png"></td>
<td><img src="https://i.ibb.co/RS4tbZs/00135-3615254454.png"></td>
<td><img src="https://i.ibb.co/mHHqBKk/00134-3615254454.png"></td>
</tr>
</table>

<table>
<tr>
<td>Original</td>
<td>Alignment center</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/hmSG5SK/00074-2037889107.png"></td>
<td><img src="https://i.ibb.co/7kPycZ5/00075-2037889107.png"></td>
</tr>
<tr>
<td>Alignment bottom</td>
<td>Alignment bottom-left</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/2gPCbr4/00076-2037889107.png"></td>
<td><img src="https://i.ibb.co/x7T91QH/00080-2037889107.png"></td>
</tr>
</table>


Exemplo de Resize (Resize sample)

<img src="https://i.ibb.co/G07KG6M/resize-00008-4017585008.png">













## Refiner

Realiza um redesenho adicional da imagem criada no txt2img.
Válido mesmo para txt2img + hires.fix.

O refiner funciona antes de detalhar a imagem, semelhante ao 
funcionamento combinado do hires.fix + refiner do sd-webui.



<table>
<tr>
<td>txt2img(512x768)</td>
<td>txt2img + hires.fix(800x1200)</td>
<td>txt2img + hires.fix + refiner(1200x1800)</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/JCxXc9D/resize-00268-767037284.png"></td>
<td><img src="https://i.ibb.co/zR3nWKt/resize-00269-767037284.png"></td>
<td><img src="https://i.ibb.co/R21B0fr/resize-00270-767037284.png"></td>
</tr>
</table>


(O exemplo acima foi redimensionado para ter o mesmo tamanho.)

Pode ser processado em 3 etapas, como no exemplo acima, ou sem a fase hires.fix, 
apenas com o refiner redimensionando.



<p>
<img src="https://i.ibb.co/GFjgJ5B/refiner.png">
</p>

#### Ativar refiner (Enable refiner)

Habilita o uso do refiner.

#### CheckPoint

Especifica o checkpoint a ser usado para redesenhar com o refiner.

#### Usar este checkpoint para detalhamento (Use this checkpoint for detailing)

Aplica o checkpoint especificado para o detalhamento.

#### Prompt

Especifica o prompt a ser usado pelo refiner ao redesenhar a imagem.
Se vazio, é igual ao prompt principal; se preenchido, ignora o prompt principal.
Se contiver a string #!org!#, substitui o prompt principal.

#### Prompt negativo (Negative prompt)

Especifica o prompt negativo a ser usado pelo refiner ao redesenhar a imagem.

#### Método de Sampling (Sampling method)

Pode especificar o sampler a ser usado pelo refiner.
(Recomendado Euler A)

#### Upscaler

Especifica o upscaler a ser usado pelo refiner ao redimensionar a imagem.

#### Etapas de sampling do Refiner (Refiner sampling steps)

Especifica as etapas de sampling a serem usadas pelo refiner.
(Recomendado 20)

#### Escala CFG do Refiner (Refiner CFG scale)

Especifica o valor da escala CFG a ser usado pelo refiner.
Não suporta dynamic threshold.

#### Força de denoising do Refiner (Refiner denoising strength)

Especifica a força de denoising a ser usada pelo refiner.
(Recomendado 0.4)

#### Escala do Refiner (Refiner scale)

O refiner redimensiona a imagem de acordo com o valor de escala especificado.
Ignorado se refiner width ou refiner height estiverem definidos.

#### Largura do Refiner (Refiner width)

Força a largura da imagem para o valor especificado.

#### Altura do Refiner (Refiner height)

Força a altura da imagem para o valor especificado.

<br>
<br>
<br>
