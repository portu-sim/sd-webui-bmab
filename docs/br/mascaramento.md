
# Mascaramento Automático (Auto Masking)

## Detectar (Detect)

* Adicione uma foto de pessoa no Img2Img.
* Selecione `Just resize`.
* Para obter bons resultados, aumente o tamanho durante o resize.
No exemplo, foi redimensionado de `512x768` para `800x1200`.
* Defina o `Denoising Strength` entre `0.4` e `0.6`.
Dependendo do modelo, reduza o valor se o fundo mudar muito.

<p>
<img src="https://i.ibb.co/VpRpN5m/2023-08-16-1-47-31.png">
</p>

* Ative o BMAB.
* Marque `Process before img2img`.
* Marque `Detect enabled` e insira no prompt algo como `person, 1girl, human... qualquer coisa`.

<p>
<img src="https://i.ibb.co/k525fb6/2023-08-16-1-47-45.png">
</p>

* Para obter bons resultados, é importante que a imagem combinada seja bem descrita no prompt.
* Se o prompt estiver vazio ou inadequado, a imagem resultante ficará longe do desejado, por isso o prompt é muito importante.

<p>
<img src="https://i.ibb.co/V0yKJdS/00015-188939271.png">
</p>

