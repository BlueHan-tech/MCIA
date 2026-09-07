"""
健康常模预计算脚本
提前计算好标准化参数和健康常模统计，保存为文件

【路径说明】
- 程序位置: F:/A-SCI/B-SCI/
- 数据位置: F:/A-SCI/Ninapro-DB2 和 F:/A-SCI/Ninapro-DB3
- 输出位置: F:/A-SCI/B-SCI/precomputed_params/
"""

import numpy as np
import json
from pathlib import Path
from ninapro_data_loader import NinaProDataLoader


class HealthyNormPrecomputer:
    """健康常模预计算器"""

    def __init__(self, db2_path, output_dir='./precomputed_params'):
        self.db2_path = db2_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = NinaProDataLoader(
            db2_path=db2_path,
            db3_path='',  # 不需要
            fs=2000
        )

    def compute_standardization_params(self, subject_ids, exercises=[1, 2, 3]):
        """
        计算标准化参数 (均值和标准差)
        使用Welford's在线算法，避免内存溢出

        Args:
            subject_ids: 被试ID列表
            exercises: 练习编号

        Returns:
            mean, std: (n_channels,) 均值和标准差
        """
        print("\n" + "="*70)
        print("计算标准化参数（均值和标准差）")
        print("="*70)

        n_channels = 12
        count = 0
        mean = np.zeros(n_channels, dtype=np.float64)
        M2 = np.zeros(n_channels, dtype=np.float64)

        for subject_id in subject_ids:
            try:
                print(f"  处理被试 S{subject_id}...")
                data = self.data_loader.load_db2_subject(subject_id, exercises)

                # 预处理（仅滤波）
                emg_filtered = self.data_loader.bandpass_filter(data['emg'])
                emg_filtered = self.data_loader.notch_filter(emg_filtered)

                # 在线更新均值和方差
                n_samples = len(emg_filtered)
                for i, sample in enumerate(emg_filtered):
                    count += 1
                    delta = sample - mean
                    mean += delta / count
                    delta2 = sample - mean
                    M2 += delta * delta2

                    # 每10000个样本打印进度
                    if (i + 1) % 10000 == 0:
                        print(f"    进度: {i+1}/{n_samples}", end='\r')

                print(f"    完成: {n_samples} 个样本")

                # 释放内存
                del data, emg_filtered

            except Exception as e:
                print(f"    ❌ 跳过被试 S{subject_id}: {e}")
                continue

        # 计算最终的标准差
        if count > 1:
            variance = M2 / (count - 1)
            std = np.sqrt(variance)

            mean = mean.astype(np.float32)
            std = std.astype(np.float32)

            print(f"\n✓ 计算完成！")
            print(f"  总样本数: {count:,}")
            print(f"  均值范围: [{mean.min():.6f}, {mean.max():.6f}]")
            print(f"  标准差范围: [{std.min():.6f}, {std.max():.6f}]")

            return mean, std
        else:
            raise ValueError("样本数不足")

    def compute_healthy_norm_stats(self, subject_ids, exercises=[1], max_samples_per_subject=10000):
        """
        计算健康常模统计（RMS能量的均值和标准差）
        用于生成静态掩码的P_norm维度

        Args:
            subject_ids: 被试ID列表（建议前10个）
            exercises: 练习编号
            max_samples_per_subject: 每个被试最大采样数

        Returns:
            norm_mean, norm_std: (n_channels,) RMS能量的均值和标准差
        """
        print("\n" + "="*70)
        print("计算健康常模统计（用于P_norm维度）")
        print("="*70)

        rms_per_channel = []

        for subject_id in subject_ids:
            try:
                print(f"  处理被试 S{subject_id}...")
                data = self.data_loader.load_db2_subject(subject_id, exercises)

                # 预处理
                emg_preprocessed, _ = self.data_loader.preprocess_emg(data['emg'])

                # 随机采样
                n_samples = min(max_samples_per_subject, len(emg_preprocessed))
                sample_indices = np.random.choice(len(emg_preprocessed), n_samples, replace=False)
                emg_sampled = emg_preprocessed[sample_indices]

                # 计算RMS能量
                rms = np.sqrt(np.mean(emg_sampled ** 2, axis=0))
                rms_per_channel.append(rms)

                print(f"    完成: 采样 {n_samples} 个样本，RMS范围=[{rms.min():.4f}, {rms.max():.4f}]")

                # 释放内存
                del data, emg_preprocessed, emg_sampled

            except Exception as e:
                print(f"    ❌ 跳过被试 S{subject_id}: {e}")
                continue

        if len(rms_per_channel) > 0:
            rms_matrix = np.array(rms_per_channel)  # (n_subjects, n_channels)

            norm_mean = np.mean(rms_matrix, axis=0).astype(np.float32)
            norm_std = np.std(rms_matrix, axis=0).astype(np.float32)

            print(f"\n✓ 计算完成！")
            print(f"  被试数: {len(rms_per_channel)}")
            print(f"  常模均值范围: [{norm_mean.min():.6f}, {norm_mean.max():.6f}]")
            print(f"  常模标准差范围: [{norm_std.min():.6f}, {norm_std.max():.6f}]")

            return norm_mean, norm_std
        else:
            raise ValueError("没有成功处理任何被试")

    def save_precomputed_data(self, scaler_mean, scaler_std, norm_mean, norm_std, subject_ids):
        """
        保存预计算的数据

        Args:
            scaler_mean: 标准化均值
            scaler_std: 标准化标准差
            norm_mean: 健康常模均值
            norm_std: 健康常模标准差
            subject_ids: 使用的被试ID列表
        """
        # 保存为.npy格式
        np.save(self.output_dir / 'scaler_mean.npy', scaler_mean)
        np.save(self.output_dir / 'scaler_std.npy', scaler_std)
        np.save(self.output_dir / 'norm_mean.npy', norm_mean)
        np.save(self.output_dir / 'norm_std.npy', norm_std)

        # 保存元信息
        metadata = {
            'subject_ids': subject_ids,
            'n_subjects': len(subject_ids),
            'n_channels': len(scaler_mean),
            'scaler_mean': scaler_mean.tolist(),
            'scaler_std': scaler_std.tolist(),
            'norm_mean': norm_mean.tolist(),
            'norm_std': norm_std.tolist(),
            'description': {
                'scaler_mean': '标准化参数 - 均值',
                'scaler_std': '标准化参数 - 标准差',
                'norm_mean': '健康常模统计 - RMS能量均值',
                'norm_std': '健康常模统计 - RMS能量标准差'
            }
        }

        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("✓✓✓ 预计算数据已保存 ✓✓✓")
        print("="*70)
        print(f"保存位置: {self.output_dir.absolute()}")
        print(f"\n生成的文件:")
        print(f"  - scaler_mean.npy  (标准化均值)")
        print(f"  - scaler_std.npy   (标准化标准差)")
        print(f"  - norm_mean.npy    (健康常模均值)")
        print(f"  - norm_std.npy     (健康常模标准差)")
        print(f"  - metadata.json    (元信息)")
        print(f"\n使用方法:")
        print(f"  在训练脚本中调用: system.load_precomputed_params(r'{self.output_dir.absolute()}')")


def main():
    """主函数"""

    # ========== 【重要】路径配置 ==========
    # 程序目录: F:\A-SCI\B-SCI\
    # 数据目录: F:\A-SCI\Ninapro-DB2 和 F:\A-SCI\Ninapro-DB3

    DB2_PATH = r'D:\code\data\Ninapro-DB2'
    OUTPUT_DIR = r'D:\code\A_clean\A_king-main\precomputed_params'  # 保存在程序目录下

    # 【重要修正】使用全部40个健康被试计算常模
    # 健康常模是统计参考值，不是训练数据，所以应该用尽可能多的数据
    # 这样可以得到更稳定、更可靠的参考范围
    ALL_HEALTHY_SUBJECTS = list(range(1, 41))  # 全部40个健康被试

    # 但是标准化参数（用于预处理训练数据）仍然只用训练集被试
    # 避免测试集数据泄露
    TRAIN_SUBJECT_IDS = [1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33,
                         34, 35, 36, 37, 38, 40]  # 32个训练被试

    print("="*70)
    print("健康常模预计算脚本")
    print("="*70)
    print(f"\n路径配置:")
    print(f"  程序目录: D:\\code\\A_clean\\A_king-main\\")
    print(f"  DB2路径: {DB2_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"\n数据配置:")
    print(f"  标准化参数: 使用 {len(TRAIN_SUBJECT_IDS)} 个训练被试")
    print(f"  健康常模: 使用全部 {len(ALL_HEALTHY_SUBJECTS)} 个健康被试")
    print(f"\n说明:")
    print(f"  - 标准化参数只用训练集（避免数据泄露）")
    print(f"  - 健康常模用全部数据（获得更稳定的参考值）")

    # 初始化预计算器
    precomputer = HealthyNormPrecomputer(DB2_PATH, OUTPUT_DIR)

    # 1. 计算标准化参数（使用所有训练被试）
    print("\n" + "="*70)
    print("[步骤 1/2] 计算标准化参数")
    print("="*70)
    print("预计耗时: 3-5分钟")
    scaler_mean, scaler_std = precomputer.compute_standardization_params(
        subject_ids=TRAIN_SUBJECT_IDS,
        exercises=[1, 2, 3]  # 使用所有练习
    )

    # 2. 计算健康常模统计（使用全部40个被试）
    print("\n" + "="*70)
    print("[步骤 2/2] 计算健康常模统计")
    print("="*70)
    print("预计耗时: 5-8分钟")
    print("使用全部40个健康被试，获得更稳定的参考值")
    norm_mean, norm_std = precomputer.compute_healthy_norm_stats(
        subject_ids=ALL_HEALTHY_SUBJECTS,  # 【修正】使用全部40个
        exercises=[1],  # 只用练习1，节省时间
        max_samples_per_subject=10000
    )

    # 3. 保存所有预计算数据
    precomputer.save_precomputed_data(
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        norm_mean=norm_mean,
        norm_std=norm_std,
        subject_ids=TRAIN_SUBJECT_IDS
    )

    print("\n" + "="*70)
    print("完成！")



if __name__ == "__main__":
    main()
