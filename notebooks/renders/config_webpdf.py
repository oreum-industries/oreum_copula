# config_webpdf.py
# per https://nbconvert.readthedocs.io/en/latest/config_options.html
# gets used alongside ~/.jupyter/jupyter_nbconvert_config.py
# which you can create from this dir as `$> jupyter nbconvert --generate-config`
# run this from inside notebooks/ dir:
# $> jupyter nbconvert --config renders/config_webpdf.py
c = get_config()
c.NbConvertApp.export_format = "webpdf"
c.FilesWriter.build_directory = "renders"  # output-dir
c.NbConvertApp.notebooks = ["000_Intro.ipynb"]
c.WebPDFExporter.embed_images = True
c.WebPDFExporter.exclude_input = True
c.WebPDFExporter.exclude_input_prompt = True
c.WebPDFExporter.exclude_output_prompt = True
# c.WebPDFExporter.allow-chromium-download = True   # not needed?
