

# Pós-processamento (Post-processing)

## Redimensionar (Resize)

### Redimensionar por pessoa (Resize by person)

É uma função que ajusta a proporção da pessoa mais alta na imagem em relação à altura da imagem se essa proporção ultrapassar o valor configurado.
Se o valor configurado for 0,90 e a proporção da altura total da pessoa em relação à altura da imagem for 0,95,
o fundo será estendido para que a proporção da pessoa seja de 0,90.
O fundo é estendido para a esquerda, direita e parte superior.


Essa função oferece duas formas de uso, como segue:

#### Inpaint

Utiliza o mesmo método do Face Detailing, expandindo as bordas da imagem após a geração completa da imagem.
Como apenas as bordas são expandidas sem alterar a imagem já gerada, é possível verificar de forma intuitiva.
Recomendado como o método mais rápido e eficaz.

#### Inpaint + lama

É semelhante ao Inpaint, mas utiliza o ControlNet no BMAB para operar com Inpaint+lama.
Após a geração da imagem, antes de iniciar o detalhamento, o img2img é usado para expandir o fundo, fazendo com que a pessoa pareça menor na imagem.

<table>
<tr>
<td><img src="https://i.ibb.co/X7P8kmH/00012-2131442442.jpg"></td>
<td><img src="https://i.ibb.co/Zc1nTz7/00022-2131442442.jpg"></td>
</tr>
<tr>
<td><img src="https://i.ibb.co/d0rSW0q/00013-2131442443.jpg"></td>
<td><img src="https://i.ibb.co/vHh2940/00023-2131442443.jpg"></td>
</tr>
<tr>
<td><img src="https://i.ibb.co/4jNJQ3b/00016-2131442446.jpg"></td>
<td><img src="https://i.ibb.co/HdcMpTH/00026-2131442446.jpg"></td>
</tr>
</table>

Esses dois métodos apenas reduzem a imagem gerada sem danificá-la.
Essa é a diferença em relação ao Resize intermediate.


## Ampliador (Upscaler)

Após a conclusão da imagem, ela é ampliada utilizando um upscaler.


#### Habilitar ampliação na etapa final (Enable upscale at final stage)

Executa a ampliação após a conclusão da geração da imagem.
Se a imagem for gerada em 960x540 e o hires.fix for x2, resultará em uma imagem 1920x1080,
e se o Upscale for x2, resultará em uma imagem 4K.

#### Detalhamento após ampliação (Detailing after upscale)

Se esta opção estiver habilitada, o detalhamento mencionado anteriormente de Pessoa, Rosto e Mão será realizado após a ampliação.

#### Proporção de ampliação (Upscale Ratio)

Decide o quanto a imagem será ampliada.


