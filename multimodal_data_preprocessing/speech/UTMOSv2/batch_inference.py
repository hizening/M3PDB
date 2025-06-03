import utmosv2
model = utmosv2.create_model(pretrained=True)
mos = model.predict(input_dir="/path/to/wav/dir/")