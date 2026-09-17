"""
数据检查脚本
检查NinaPro DB2数据是否正常加载
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

def check_ninapro_data():
    """检查NinaPro数据"""
    
    db2_path = Path(r'D:\code\data\Ninapro-DB2')
    
    print("=" * 70)
    print("NinaPro DB2 数据检查")
    print("=" * 70)
    
    # 检查·径
    print(f"\n1. 检查数据路径...")
    print(f"   DB2 路径：{db2_path}")
    print(f"   路径存在：{db2_path.exists()}")
    
    if not db2_path.exists():
        print(f"   ❌ 错误：路径不存在！")
        return
    
    # 列出可用被试
    print(f"\n2. 检查可用被试...")
    subject_dirs = list(db2_path.glob("DB2_s*"))
    print(f"   找到 {len(subject_dirs)} 个被试文件夹")
    if subject_dirs:
        print(f"   示例：{subject_dirs[0].name}")
    
    # 检查第һ个被试的数据
    test_subject = 1
    subject_dir = db2_path / f"DB2_s{test_subject}"
    
    print(f"\n3. 检查被试 S{test_subject} 的数据...")
    print(f"   被试目录：{subject_dir}")
    print(f"   目录存在：{subject_dir.exists()}")
    
    if not subject_dir.exists():
        print(f"   ❌ 错误：被试目录不存在！")
        return
    
    # 列出文件
    mat_files = list(subject_dir.glob("*.mat"))
    print(f"   找到 {len(mat_files)} 个.mat文件")
    for f in mat_files[:3]:
        print(f"     - {f.name}")
    
    # 加载练习 1 的数据
    mat_file = subject_dir / f"S{test_subject}_E1_A1.mat"
    print(f"\n4. 加载文件: {mat_file.name}")
    print(f"   文件存在: {mat_file.exists()}")
    
    if not mat_file.exists():
        print(f"   ❌ 错误：文件不存在！")
        return
    
    try:
        data = sio.loadmat(mat_file)
        print(f"   ✓ 文件加载成功")
        
        # 检查数据结构
        print(f"\n5. 检查数据结构...")
        print(f"   数据字段: {list(data.keys())}")
        
        # 检查EMG数据
        if 'emg' in data:
            emg = data['emg']
            print(f"\n6. EMG数据详情:")
            print(f"   形状：{emg.shape}")
            print(f"   数据类型: {emg.dtype}")
            print(f"   范围：[{emg.min():.6f}, {emg.max():.6f}]")
            print(f"   均值：{emg.mean():.6f}")
            print(f"   标准差：{emg.std():.6f}")
            print(f"   中位数：{np.median(emg):.6f}")
            
            # 显ʾǰ几行数据
            print(f"\n7. 前 5 行数据示例：")
            print(emg[:5, :3])  # 前 5 行、前 3 个通道
            
            # 检查是否有异常ֵ
            print(f"\n8. 数据质量检查:")
            print(f"   NaN数量: {np.isnan(emg).sum()}")
            print(f"   Inf数量: {np.isinf(emg).sum()}")
            print(f"   零值比例：{(emg == 0).sum() / emg.size * 100:.2f}%")
            
            # 百分位数分布
            print(f"\n9. 数据分布:")
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            for p in percentiles:
                val = np.percentile(emg, p)
                print(f"   {p:2d}% 分位数：{val:.6f}")
            
            # 检查滤波后的数据
            print(f"\n10. 模拟滤波（简单检查）：")
            from scipy import signal
            
            fs = 2000
            nyq = 0.5 * fs
            low = 20 / nyq
            high = 450 / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            
            # 只滤波一个通道作为测试
            emg_filtered = signal.filtfilt(b, a, emg[:, 0])
            
            print(f"   滤波后范围：[{emg_filtered.min():.6f}, {emg_filtered.max():.6f}]")
            print(f"   滤波后均值：{emg_filtered.mean():.6f}")
            print(f"   滤波后标准差：{emg_filtered.std():.6f}")
            
            # 检查数据单位
            print(f"\n11. 数据单位推断：")
            if abs(emg.max()) < 0.1:
                print(f"   ⚠️ 数据范围很小（<0.1），可能单位是：")
                print(f"      - 毫伏（mV），已经过预处理")
                print(f"      - 或已经归一化")
            elif abs(emg.max()) < 10:
                print(f"   ✓ 数据范围正常（<10），可能单位是毫伏（mV）")
            else:
                print(f"   数据范围较大（>10），可能单位是微伏（μV）")
            
            # 总结
            print(f"\n" + "=" * 70)
            print("诊断总结")
            print("=" * 70)
            
            if emg.std() < 0.001:
                print("⚠️ 警告：数据标准差很小（<0.001）")
                print("   可能原因：")
                print("   1. 数据已经过标准化/归一化")
                print("   2. 数据单位是 mV 而不是 μV")
                print("   3. 数据质量有问题")
                print("\n   建议：")
                print("   - 不要再次标准化，直接使用原始数据")
                print("   - 或者放大数据（×1000）后再标准化")
            elif emg.std() > 100:
                print("⚠️ 警告：数据标准差很大（>100）")
                print("   可能原因：单位是 μV")
                print("   建议：正常进行标准化")
            else:
                print("✓ 数据范围正常")
                
        else:
            print(f"   ❌ 错误：数据中没有 'emg' 字段！")
            
    except Exception as e:
        print(f"   ❌ 错误：加载文件失败！")
        print(f"   错误信息：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_ninapro_data()

