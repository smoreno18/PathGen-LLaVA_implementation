<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>PAthGen-LLaVA implementation - README</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #c9d1d9;
            background-color: #0d1117;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        h1, h2, h3 {
            color: #f0f6fc;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }
        code {
            background-color: rgba(110, 118, 129, 0.4);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #161b22;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
            border: 1px solid #30363d;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            font-size: 90%;
            color: #e6edf3;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        .highlight {
            color: #58a6ff;
        }
        .note {
            background-color: #161b22;
            border-left: 4px solid #d29922;
            padding: 10px 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>

    <h1>PAthGen-LLaVA – Inferencia sobre region de WSI</h1>
    <p>Este repositorio contiene una versión de <strong>PathGen-LLaVA</strong> adaptada para realizar inferencia sobre regiones específicas de Whole Slide Images (WSI), scripts claves:</p>
    
    <ul>
        <li><code>main.py</code> &rarr; función de carga del modelo e inferencia sobre una región de WSI</li>
        <li><code>demo.py</code> &rarr; demo de uso</li>
    </ul>

    <h2>Preparación del Modelo</h2>
    <p>Antes de construir o lanzar el contenedor, es necesario descargar los pesos del modelo <strong>PathGen-LLaVA</strong> de forma local.</p>
    
    <h3>Descarga desde Hugging Face CLI</h3>
    <pre><code># Descarga en carpeta local
huggingface-cli download jamessyx/PathGen-LLaVA --local-dir llava_dir/PathGen-LLaVA</code></pre>


    <h2>Construir imagen docker y lanzar contenedor</h2>

    <h3># Build</h3>
    <pre><code>docker build -t pathgen .</code></pre>

    <h3># Run</h3>
    <pre><code> docker run <CONFIG> quilt-llava-qllava</code></pre>

</body>
</html>