# 测试流程

环境迁移步骤与 water plant 示例见 [迁移指南](docs/迁移指南.md)。

先下载原始数据，然后转换成lerobot数据放在./datasets里面

参考
```bash
uv run dexjoco-convert \
  --task water_plant \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/water_plant \
  --output-dir datasets/water_plant \
  --n-envs 16
```

然后运行测试，检查replay是否成功
```bash
uv run pytest
```

经过转换筛选之后，剩余的episode数量可能不足100

抗锯齿的bug：只能保证环境里的图片是确定的，但是数据集里的图片是有损压缩的