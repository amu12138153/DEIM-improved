from engine.core import YAMLConfig
cfg = YAMLConfig(r'configs/deimv2/visdrone_pico.yml')
s, t = cfg.val_dataloader.dataset[0]
print(t['boxes'], '\nmax =', t['boxes'].max())  # max 应该在 0~1 之间