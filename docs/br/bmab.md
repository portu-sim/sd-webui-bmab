
# BMAB

## Funções Básicas (Basic Functions)

* Contrast: Ajuste do valor de contraste (1 não altera)
* Brightness: Ajuste do valor de brilho (1 não altera)
* Sharpeness: Ajuste do valor de nitidez (1 não altera)
* Color Temperature: Ajuste da temperatura de cor, 6500K é 0 (0 não altera)
* Noise alpha: Adiciona ruído antes do processo para aumentar os detalhes. (Valor recomendado: 0.1)
* Noise alpha at final stage: Adiciona ruído na fase final para alterar a atmosfera.


## Imagem (Imaging)

### Misturar Imagem no Img2Img (Blend Image in Img2Img)

Mistura a imagem inserida na caixa de upload com a imagem inserida no Img2Img.
Usa o valor de `Blend Alpha` para combinar as duas imagens.
A opção `Process before Img2Img` é aplicada.

### Detecção Dino (Dino detect)

Quando usando Img2Img Inpainting, não é necessário inserir uma máscara, pois o prompt do Dino detect gera automaticamente a máscara.
Ao carregar uma imagem, ela é usada como fundo e a parte inserida pelo prompt é combinada com a imagem carregada.

#### Quando usar no Img2Img

<p>
<img src="https://i.ibb.co/W5xs487/00027-3690585574.png" width="40%">
<img src="https://i.ibb.co/rk7xDSR/00467-2764185410.png" width="40%">
</p>
<p>
<img src="https://i.ibb.co/Byw3rY6/tmp3478vdur.png" width="40%">
<img src="https://i.ibb.co/7W6QhTG/00024-155186649.png" width="40%">
</p>



Primeira imagem é definida como imagem do Img2Img
Segunda imagem é definida na caixa de entrada de imagem do BMAB Imaging

Durante o processo, a terceira imagem é combinada e o resultado é obtido de acordo com o prompt.

Enabled: CHECK!!   

Contrast: 1.2   
Brightness: 0.9   
Sharpeness: 1.5

Enable dino detect: CHECK!!   
DINO detect Prompt: 1girl


#### Quando usar no Img2Img Inpaint

A máscara é gerada automaticamente de acordo com o prompt do DINO detect.

<p>
<img src="https://i.ibb.co/W5xs487/00027-3690585574.png" width="30%">
<img src="https://i.ibb.co/80qQvDv/tmpnm78iuqo.png" width="30%">
<img src="https://i.ibb.co/mRT77BM/00028-2672855487.png" width="30%">
</p>


Neste exemplo, o fundo foi alterado, então é necessário selecionar `Inpaint Not Masked` nas configurações de inpaint.
Por outro lado, se `Inpaint Masked` for selecionado, o personagem será alterado.


## Pessoa (Person)

Ao usar esta função, após o processo ser concluído, a pessoa será detectada e redesenhada.
É eficaz nos seguintes casos:

* Quando a pessoa é muito pequena em comparação com o fundo, aumentando os detalhes de roupas, rosto e todo o corpo.
* Quando gerando imagens grandes, como 4K, se a pessoa ficar pequena após o upscale, esta função ajuda a torná-la mais nítida.
* Funciona bem com Face Detailing.


<img src="https://i.ibb.co/RSrvqM1/person.png">


#### Habilitar detalhamento de pessoa para paisagem (EXPERIMENTAL) (Enable person detailing for landscape (EXPERIMENTAL))

Ativa a função de redesenhar pessoas em paisagens com mais detalhes.

#### Bloquear imagem superdimensionada (Block over-scaled image)

Quando ativada, encontra a pessoa e a amplia para redesenhar, mas se a área ampliada exceder a imagem original, o processo é interrompido.
Protege o sd-webui de travamentos ou a GPU de sobrecarga.

#### Auto ajustar escala se "Bloquear imagem superdimensionada" estiver habilitado (Auto scale if "Block over-scaled image" enabled)

Ajusta a escala para se adequar à área original da imagem se "Block over-scaled image" estiver ativado e a área ampliada for bloqueada.

#### Proporção de Ampliação (Upscale Ratio)

Amplia a pessoa encontrada de acordo com a proporção especificada para redesenhá-la com mais detalhes.

#### Força de Desnatação (Denoising Strength)

Se a pessoa for grande, 0.4 pode ser insuficiente. Aumente o valor nesses casos.

#### Máscara de Dilatação (Dilation mask)

Expande a máscara da pessoa detectada.

#### Escala CFG (CFG Scale)

Valor do `CFG scale` usado ao redesenhar a pessoa.

#### Limite de área grande de pessoa (Large person area limit)

Se a área ocupada pela pessoa na imagem exceder este valor, o redesenho não será realizado.
Não é necessário redesenhar se a pessoa já for grande o suficiente.

#### Limite (Limit)

Se houver muitas pessoas na imagem, conta a partir da maior e para ao exceder o valor definido.


<img src="https://i.ibb.co/n8PmL3P/00057-2574875327.jpg">
<img src="https://i.ibb.co/r2fdSmJ/00399-1097195856.png">


## Rosto (Face)

### Detalhamento de Rosto (Face Detailing)

Ao usar esta função, após o processo ser concluído, corrige o rosto usando o `After Detailer (AD)`
ou `Detection Detailer (DD)`.
Se configurado para funcionar após o AD ou DD, os resultados podem não ser satisfatórios.

<img src="https://i.ibb.co/frx85BR/face.png">

É possível especificar prompts separadamente para até 5 personagens.

#### Habilitar detalhamento de rosto (Enable face detailing)

Ativa ou desativa a função `face detailing`.

#### Habilitar detalhamento de rosto antes do hires.fix (EXPERIMENTAL) (Enable face detailing before hires.fix (EXPERIMENTAL))

Executa a função `face detailing` antes do hires.fix no processo txt2img.
Como a correção do rosto é feita antes do upscale, a qualidade da imagem final é melhor.
Entretanto, aumenta a carga e a mudança na imagem pode ser significativa.

#### Ordenar detalhamento de rosto por (Face detailing sort by)

Decide a ordem de Detailing se houver várias pessoas na imagem.

<img src="https://i.ibb.co/DR8g34t/00037-3214376443.png">
<img src="https://i.ibb.co/4JXdkpT/00036-3214376443.png">

Pode ser pela posição esquerda, direita ou tamanho, sendo o padrão a ordem pelo valor de Score.

#### Limite (Limit)

Determina quantas pessoas serão detalhadas na ordem definida anteriormente.
`Limit` com valor 1 significa que apenas 1 será detalhado.

#### Substituir Parâmetros (Override Parameters)

* Denoising Strength
* CFG Scale
* Width
* Height
* Steps
* Mask Blur

Usa os valores definidos na UI em vez dos valores padrão.

#### Área de Inpaint (Inpaint Area)

Decide se todo o rosto ou apenas a face será redesenhado. Não é recomendado redesenhar tudo.

#### Apenas preenchimento mascarado, pixels (Only masked padding, pixels)

Use o valor padrão.

#### Dilatação (Dilation)

Aumenta o tamanho da máscara do rosto detectado.

#### Limite de caixa (Box threshold)

Determina o valor de detecção do detector. Valores menores que 0.35 são excluídos como não sendo face.
Se usando YOLO, substitui a confiança.

**Dicas para bons resultados**

* Remova lora, textual inversion, etc., relacionados ao rosto do prompt. Óculos de sol, etc., são irrelevantes.
* Adicione diferentes lora, textual inversion, etc., para cada rosto no arquivo de configuração.
* Muitos lora, TI no prompt podem reduzir a liberdade de criação da imagem.
* Lora compartilhados entre todos os personagens na imagem são permitidos.



## Mão (Hand)

### Detalhamento de Mão (EXPERIMENTAL) (Hand Detailing (EXPERIMENTAL))

Corrige partes mal desenhadas das mãos.
Encontra automaticamente as mãos na imagem gerada e redesenha essas partes.
Entretanto, mesmo redesenhando, pode não ficar perfeito, apenas detalhado.

<img src="https://i.ibb.co/fxQh9ZN/hand.png">

#### Habilitar detalhamento de mão (Enable hand detailing)

Ativa a função de detalhamento de mãos.

#### Bloquear imagem superdimensionada (Block over-scaled image)

Esta função encontra as mãos e amplia para redesenhá-las.
Se a área a ser redesenhada exceder a imagem original, o trabalho é interrompido.
Reduza o `Upscale Ratio` ou desative esta função, mas desativar pode sobrecarregar a GPU ao redesenhar uma imagem muito grande.

#### Método (Method)
* subframe: Encontra as mãos, rosto/cabeça, e redesenha a parte superior do corpo.
* each hand: Encontra as mãos e redesenha uma área 3 vezes maior ao redor de cada mão.
* each hand inpaint: Encontra as mãos e redesenha uma área 3 vezes maior ao redor de cada mão, mas pode
ficar muito distorcido. Recomenda-se usar subframe se a forma estiver muito alterada.
* at once: Redesena todas as mãos encontradas de uma vez.


#### Prompt

Não insira no `subframe`.
Insira prompts relacionados às mãos para each hand, each hand inpaint.

#### Prompt Negativo (Negative Prompt)

Não insira no subframe.
Insira prompts negativos relacionados às mãos para each hand, each hand inpaint.

#### Força da Remoção de Ruído (Denoising Strength)

Valor do `Denoising Strength` ao redesenhar.
* subframe: recomendado 0.4
* outros: recomendado acima de 0.55

#### Escala CFG (CFG Scale)

Valor de CFG Scale ao redesenhar.

#### Proporção de Ampliação (Upscale Ratio)
Determina quanto ampliar ao redesenhar a parte superior do corpo / área ao redor das mãos.
Aumentar muito não garante melhores resultados.
* subframe: 2.0
* outros: 2.0~4.0

#### Limite de caixa (Box Threshold)

Se as mãos não forem encontradas, reduza este valor para aumentar a chance de detecção,
mas também aumenta a chance de erros.

#### Dilatação de caixa (Box Dilation)

Determina o tamanho da borda ao redor da caixa detectada (incluindo as mãos). (somente para subframe)

#### Área de Inpaint (Inpaint Area)

Decide se a caixa inteira detectada ou apenas as mãos serão redesenhadas.
Redesenhar apenas as mãos pode alterar a forma inesperadamente.

#### Apenas preenchimento mascarado (Only masked padding)

Determina o quanto preencher o espaço interno das mãos detectadas. Não é necessário mudar.

#### Parâmetro Adicional (Additional Parameter)

Atualmente não disponível, mas opções avançadas serão fornecidas no futuro.



## ControlNet

Usa ControlNet para adicionar ruído à imagem e aumentar os detalhes.
Insere uma imagem de ruído gaussiano no modelo Lineart do ControlNet para 
adicionar detalhes variados e complexos ao resultado.

#### Força do Ruído (Noise Strength)

Define a intensidade do ruído. (Recomendado: 0.4)

#### Início (Begin)

Ponto de início da etapa de amostragem.

#### Final (End)

Ponto de término da etapa de amostragem.

Geralmente, recomenda-se 0.4, 0, 0.4. Se a imagem estiver excessivamente desenhada, 
recomenda-se 0.2, 0, 0.4. Usar um refinador pode estabilizar a imagem se estiver excessivamente desenhada.

Todas as imagens abaixo usam o mesmo seed.
<table>
<tr>
<td>Imagem original</td>
<td>0.4</td>
<td>0.7</td>
</tr>
<tr>
<td><img src="https://i.ibb.co/ypRrwmN/00007-51151519.jpg"></td>
<td><img src="https://i.ibb.co/j54HfHF/00009-51151519.jpg"></td>
<td><img src="https://i.ibb.co/MsgCZS3/00008-51151519.jpg"></td>
</tr>
</table>






<br>
<br>
<br>
