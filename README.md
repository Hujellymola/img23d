# 3D Model Texture Completion Pipeline

这是一个完整的3D模型纹理补全pipeline，集成了PartField分割、HoloPart几何补全和InTeX纹理修复。

## 系统架构

```
PartField输出 → 格式转换 → HoloPart补全 → 纹理传递 → InTeX修复 → 最终组装
```

## 脚本说明

### 1. `convert_partfield_to_holopart.py`
**环境**: partfield conda环境  
**功能**: 将PartField聚类结果转换为HoloPart输入格式

```bash
python convert_partfield_to_holopart.py \
    --labels-npy /path/to/labels.npy \
    --original-glb /path/to/original.glb \
    --output-dir ./step1_conversion
```

**输出**:
- `parts_for_holopart.glb`: HoloPart输入文件
- `conversion_metadata.json`: 转换元数据
- `original_texture.png`: 原始纹理备份

### 2. `run_holopart_completion.py`
**环境**: holopart conda环境  
**功能**: 执行HoloPart几何补全

```bash
python run_holopart_completion.py \
    --input-scene ./step1_conversion/parts_for_holopart.glb \
    --output-dir ./step2_completion \
    --batch-size 4
```

**输出**:
- `completed_scene.glb`: 补全后的完整场景
- `part_XX_completed.obj`: 各部件补全后的几何体
- `completion_metadata.json`: 补全元数据

### 3. `compute_texture_correspondence.py`
**环境**: 基础Python环境  
**功能**: 计算原始纹理到补全几何的对应关系

```bash
python compute_texture_correspondence.py \
    --conversion-metadata ./step1_conversion/conversion_metadata.json \
    --completion-metadata ./step2_completion/completion_metadata.json \
    --output-dir ./step3_correspondence
```

**输出**:
- `part_XX/projected_texture.png`: 投影纹理
- `part_XX/inpaint_mask.png`: 修复遮罩
- `correspondence_metadata.json`: 对应关系元数据

### 4. `prepare_intex_input.py`
**环境**: intex conda环境  
**功能**: 准备InTeX输入格式

```bash
python prepare_intex_input.py \
    --correspondence-metadata ./step3_correspondence/correspondence_metadata.json \
    --completion-metadata ./step2_completion/completion_metadata.json \
    --global-prompt "a detailed 3D model with realistic textures" \
    --output-dir ./step4_intex_input
```

**输出**:
- `part_XX/textured_mesh.glb`: 带纹理的网格
- `part_XX/masked_texture_raw.npy`: InTeX格式的遮罩纹理
- `intex_batch_list.json`: 批处理列表

### 5. `run_intex_inpainting.py`
**环境**: intex conda环境  
**功能**: 执行InTeX纹理修复

```bash
python run_intex_inpainting.py \
    --batch-list ./step4_intex_input/intex_batch_list.json \
    --output-dir ./step5_inpainting
```

**输出**:
- `part_XX_inpainted.glb`: 修复后的网格
- `part_XX_final_texture.png`: 最终纹理
- `inpainting_results.json`: 修复结果元数据

### 6. `assemble_final_model.py`
**环境**: 基础Python环境  
**功能**: 组装最终完整模型

```bash
python assemble_final_model.py \
    --inpainting-results ./step5_inpainting/inpainting_results.json \
    --output-path ./final_model.glb \
    --atlas-size 4096
```

**输出**:
- `final_model.glb`: 最终完整模型
- `final_model_atlas.png`: 统一纹理图集
- `final_model_metadata.json`: 组装元数据

## 完整Pipeline示例

```bash
# 步骤1: 转换PartField输出
conda activate partfield
python convert_partfield_to_holopart.py \
    --labels-npy /path/to/partfield/labels.npy \
    --original-glb /path/to/original.glb \
    --output-dir ./pipeline_output/step1_conversion

# 步骤2: HoloPart补全
conda activate holopart
python run_holopart_completion.py \
    --input-scene ./pipeline_output/step1_conversion/parts_for_holopart.glb \
    --output-dir ./pipeline_output/step2_completion

# 步骤3: 计算纹理对应关系
conda activate base  # 或其他基础环境
python compute_texture_correspondence.py \
    --conversion-metadata ./pipeline_output/step1_conversion/conversion_metadata.json \
    --completion-metadata ./pipeline_output/step2_completion/completion_metadata.json \
    --output-dir ./pipeline_output/step3_correspondence

# 步骤4: 准备InTeX输入
conda activate intex
python prepare_intex_input.py \
    --correspondence-metadata ./pipeline_output/step3_correspondence/correspondence_metadata.json \
    --completion-metadata ./pipeline_output/step2_completion/completion_metadata.json \
    --global-prompt "realistic 3D object with detailed surface textures" \
    --output-dir ./pipeline_output/step4_intex_input

# 步骤5: InTeX修复
python run_intex_inpainting.py \
    --batch-list ./pipeline_output/step4_intex_input/intex_batch_list.json \
    --output-dir ./pipeline_output/step5_inpainting

# 步骤6: 组装最终模型
conda activate base
python assemble_final_model.py \
    --inpainting-results ./pipeline_output/step5_inpainting/inpainting_results.json \
    --output-path ./final_complete_model.glb
```

## 核心特性

- **模块化设计**: 每个脚本职责单一，便于调试
- **环境隔离**: 支持不同conda环境，避免依赖冲突
- **纹理保护**: 原始可见纹理得到保护，只对缺失区域进行修复
- **完整追踪**: 每个阶段都有详细的元数据记录
- **错误暴露**: 不包含fallback，问题会快速暴露

## 依赖要求

### 基础依赖 (所有环境)
```bash
pip install trimesh numpy pillow scipy scikit-learn
```

### PartField环境
```bash
# PartField特定依赖
```

### HoloPart环境  
```bash
# HoloPart特定依赖
pip install torch torchvision pymeshlab
```

### InTeX环境
```bash
# InTeX特定依赖
pip install torch torchvision diffusers
```

## 注意事项

1. **路径管理**: 所有脚本都使用绝对路径，确保在正确目录运行
2. **内存管理**: HoloPart和InTeX处理大模型时需要充足GPU内存
3. **纹理质量**: 最终纹理质量取决于原始纹理质量和InTeX参数调优
4. **调试支持**: 每个步骤都会保存中间结果，便于问题定位

## 故障排除

如果某个步骤失败：
1. 检查对应conda环境是否正确激活
2. 查看输出的错误信息和日志
3. 检查中间文件是否正确生成
4. 验证输入文件格式和路径

这个pipeline设计确保了每个步骤的独立性和可调试性，同时保持了纹理传递的一致性。
