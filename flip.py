import os
import numpy as np
from osgeo import gdal

def image_flip_image_leftright(input_path, output_path):

    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open {input_path}")
        return

    band_count = ds.RasterCount
    if band_count != 6:
        print(f"{input_path} does not have exactly 6 bands.")
        return

    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, x_size, y_size, band_count, gdal.GDT_Float32)
    if out_ds is None:
        print(f"Failed to create output file {output_path}")
        return

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    for i in range(band_count):
        band = ds.GetRasterBand(i + 1)
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(band.ReadAsArray()[:, ::-1])

    out_ds.FlushCache()
    out_ds = None
    ds = None

def image_flip_image_vertical(input_path, output_path):
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open {input_path}")
        return

    band_count = ds.RasterCount
    if band_count != 6:
        print(f"{input_path} does not have exactly 6 bands.")
        return


    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    output_path = output_path + ''

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, x_size, y_size, band_count, ds.GetRasterBand(1).DataType)
    if out_ds is None:
        print(f"Failed to create output file {output_path}")
        return

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    for i in range(band_count):
        band = ds.GetRasterBand(i + 1)
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(band.ReadAsArray()[::-1, :])

    out_ds.FlushCache()
    out_ds = None
    ds = None


def image_process_directory(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_dir, filename)

            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, name + '_flip_horizontal' + ext)
            image_flip_image_leftright(input_path, output_path)
            output_path = os.path.join(output_dir, name + '_flip_vertical' + ext)
            image_flip_image_vertical(input_path, output_path)

            print(f"Processed {filename}")

def flip_image_leftright(input_path, output_path):
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open {input_path}")
        return

    band_count = ds.RasterCount

    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, x_size, y_size, band_count, gdal.GDT_Float32)
    if out_ds is None:
        print(f"Failed to create output file {output_path}")
        return

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    for i in range(band_count):
        band = ds.GetRasterBand(i + 1)
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(band.ReadAsArray()[:, ::-1])

    out_ds.FlushCache()
    out_ds = None
    ds = None

def flip_image_vertical(input_path, output_path):
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open {input_path}")
        return

    band_count = ds.RasterCount

    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    output_path = output_path + ''

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, x_size, y_size, band_count, ds.GetRasterBand(1).DataType)
    if out_ds is None:
        print(f"Failed to create output file {output_path}")
        return

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    for i in range(band_count):
        band = ds.GetRasterBand(i + 1)
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(band.ReadAsArray()[::-1, :])

    out_ds.FlushCache()
    out_ds = None
    ds = None


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_dir, filename)

            name, ext = os.path.splitext(filename)

            output_path = os.path.join(output_dir, name + '_flip_horizontal' + ext)
            flip_image_leftright(input_path, output_path)
            output_path = os.path.join(output_dir, name + '_flip_vertical' + ext)
            flip_image_vertical(input_path, output_path)

            print(f"Processed {filename}")

if __name__ == "__main__":
    process_directory(label_input_folder, label_output_folder)
    image_process_directory(input_folder, output_folder)

