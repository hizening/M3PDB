import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import whisper


class BatchAudioProcessor:
    """批量音频文件处理类"""
    
    def __init__(self, model_name="turbo"):
        """
        初始化批量音频处理器
        
        Args:
            model_name: Whisper模型名称 (tiny, base, small, medium, large, turbo)
        """
        self.model_name = model_name
        self.model = None
        self.logger = self._setup_logger()
        
        # 支持的音频文件格式
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('BatchAudioProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self):
        """加载Whisper模型"""
        if self.model is None:
            self.logger.info(f"正在加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.logger.info("模型加载完成")
    
    def get_audio_files(self, folder_path: str) -> List[str]:
        """
        获取文件夹中所有支持的音频文件
        
        Args:
            folder_path: 音频文件夹路径
            
        Returns:
            音频文件路径列表
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        audio_files = []
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                audio_files.append(str(file_path))
        
        audio_files.sort()  # 按文件名排序
        self.logger.info(f"找到 {len(audio_files)} 个音频文件")
        return audio_files
    
    def process_single_audio(self, audio_path: str, segment_length: int = 30) -> Dict[str, Any]:
        """
        处理单个音频文件
        
        Args:
            audio_path: 音频文件路径
            segment_length: 音频片段长度（秒），0表示处理整个文件
            
        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"正在处理: {os.path.basename(audio_path)}")
            
            # 加载音频
            audio = whisper.load_audio(audio_path)
            
            results = []
            
            if segment_length > 0:
                # 分段处理
                segment_samples = segment_length * 16000  # 16kHz采样率
                total_segments = len(audio) // segment_samples + (1 if len(audio) % segment_samples > 0 else 0)
                
                for i in range(total_segments):
                    start_idx = i * segment_samples
                    end_idx = min((i + 1) * segment_samples, len(audio))
                    audio_segment = audio[start_idx:end_idx]
                    
                    # 处理音频段
                    segment_result = self._process_audio_segment(
                        audio_segment, 
                        segment_id=i,
                        start_time=start_idx / 16000,
                        end_time=end_idx / 16000
                    )
                    results.append(segment_result)
            else:
                # 处理整个音频文件
                result = self._process_audio_segment(audio, segment_id=0, start_time=0, end_time=len(audio) / 16000)
                results.append(result)
            
            # 合并结果
            combined_result = self._combine_results(results, audio_path)
            return combined_result
            
        except Exception as e:
            error_msg = f"处理音频文件失败 {audio_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'file_path': audio_path,
                'file_name': os.path.basename(audio_path),
                'status': 'error',
                'error': str(e),
                'detected_language': None,
                'language_probability': None,
                'text': None,
                'segments': []
            }
    
    def _process_audio_segment(self, audio, segment_id: int = 0, start_time: float = 0, end_time: float = 0) -> Dict[str, Any]:
        """处理音频片段"""
        # 填充或裁剪音频到固定长度（如果需要）
        audio_processed = whisper.pad_or_trim(audio)
        
        # 生成log-Mel频谱图
        mel = whisper.log_mel_spectrogram(
            audio_processed, 
            n_mels=self.model.dims.n_mels
        ).to(self.model.device)
        
        # 检测语言
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        language_prob = probs[detected_language]
        
        # 解码音频
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)
        
        return {
            'segment_id': segment_id,
            'start_time': start_time,
            'end_time': end_time,
            'detected_language': detected_language,
            'language_probability': language_prob,
            'text': result.text.strip(),
            'no_speech_prob': getattr(result, 'no_speech_prob', None)
        }
    
    def _combine_results(self, segments: List[Dict[str, Any]], audio_path: str) -> Dict[str, Any]:
        """合并多个音频段的结果"""
        if not segments:
            return {}
        
        # 找出最常见的语言
        language_counts = {}
        total_prob = 0
        
        for segment in segments:
            lang = segment['detected_language']
            prob = segment['language_probability']
            
            if lang in language_counts:
                language_counts[lang] += prob
            else:
                language_counts[lang] = prob
            total_prob += prob
        
        # 确定主要语言
        main_language = max(language_counts, key=language_counts.get)
        main_language_prob = language_counts[main_language] / total_prob if total_prob > 0 else 0
        
        # 合并文本
        combined_text = ' '.join(segment['text'] for segment in segments if segment['text'])
        
        return {
            'file_path': audio_path,
            'file_name': os.path.basename(audio_path),
            'status': 'success',
            'detected_language': main_language,
            'language_probability': main_language_prob,
            'text': combined_text,
            'segments': segments,
            'total_segments': len(segments)
        }
    
    def process_folder(self, folder_path: str, output_dir: str = None, 
                      segment_length: int = 30, save_individual: bool = True) -> Dict[str, Any]:
        """
        批量处理文件夹中的所有音频文件
        
        Args:
            folder_path: 音频文件夹路径
            output_dir: 输出目录路径，None则使用输入文件夹
            segment_length: 音频片段长度（秒），0表示处理整个文件
            save_individual: 是否保存每个文件的单独结果
            
        Returns:
            批量处理结果统计
        """
        # 确保模型已加载
        self.load_model()
        
        # 获取音频文件列表
        audio_files = self.get_audio_files(folder_path)
        
        if not audio_files:
            self.logger.warning("未找到任何支持的音频文件")
            return {'total_files': 0, 'successful': 0, 'failed': 0, 'results': []}
        
        # 设置输出目录
        if output_dir is None:
            output_dir = folder_path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 处理所有音频文件
        results = []
        successful = 0
        failed = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            self.logger.info(f"处理进度: {i}/{len(audio_files)}")
            
            result = self.process_single_audio(audio_file, segment_length)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            # 保存单个文件结果
            if save_individual and result['status'] == 'success':
                self._save_individual_result(result, output_path)
        
        # 保存汇总结果
        summary = {
            'processing_time': datetime.now().isoformat(),
            'model_used': self.model_name,
            'total_files': len(audio_files),
            'successful': successful,
            'failed': failed,
            'segment_length': segment_length,
            'results': results
        }
        
        self._save_summary_results(summary, output_path)
        
        self.logger.info(f"批量处理完成: {successful}/{len(audio_files)} 成功")
        return summary
    
    def _save_individual_result(self, result: Dict[str, Any], output_path: Path):
        """保存单个文件的处理结果"""
        file_name = Path(result['file_name']).stem
        
        # 保存为JSON格式
        json_file = output_path / f"{file_name}_transcription.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存为纯文本格式
        txt_file = output_path / f"{file_name}_transcription.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"文件: {result['file_name']}\n")
            f.write(f"检测语言: {result['detected_language']} (置信度: {result['language_probability']:.3f})\n")
            f.write(f"转录文本:\n{result['text']}\n")
    
    def _save_summary_results(self, summary: Dict[str, Any], output_path: Path):
        """保存汇总结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整JSON结果
        json_file = output_path / f"batch_transcription_summary_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式的简要结果
        csv_file = output_path / f"batch_transcription_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '状态', '检测语言', '语言置信度', '转录文本'])
            
            for result in summary['results']:
                writer.writerow([
                    result['file_name'],
                    result['status'],
                    result.get('detected_language', ''),
                    f"{result.get('language_probability', 0):.3f}",
                    result.get('text', '')[:100] + '...' if result.get('text', '') else ''
                ])
        
        self.logger.info(f"结果已保存到: {json_file}")
        self.logger.info(f"CSV摘要已保存到: {csv_file}")


def main():
    """主函数示例"""
    # 创建处理器
    processor = BatchAudioProcessor(model_name="turbo")
    
    # 设置路径
    audio_folder = "audio_files"  # 替换为你的音频文件夹路径
    output_folder = "transcription_results"  # 输出文件夹路径
    
    try:
        # 批量处理
        results = processor.process_folder(
            folder_path=audio_folder,
            output_dir=output_folder,
            segment_length=30,  # 30秒片段，设为0则处理整个文件
            save_individual=True
        )
        
        print("\n=== 处理结果汇总 ===")
        print(f"总文件数: {results['total_files']}")
        print(f"成功处理: {results['successful']}")
        print(f"处理失败: {results['failed']}")
        print(f"使用模型: {results['model_used']}")
        
        # 显示部分结果
        if results['results']:
            print("\n=== 部分转录结果 ===")
            for result in results['results'][:3]:  # 显示前3个结果
                if result['status'] == 'success':
                    print(f"\n文件: {result['file_name']}")
                    print(f"语言: {result['detected_language']} ({result['language_probability']:.3f})")
                    print(f"文本: {result['text'][:100]}...")
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()


# 快速使用示例
def quick_process_example():
    """快速处理示例"""
    processor = BatchAudioProcessor("turbo")
    
    # 一行代码批量处理
    results = processor.process_folder("./audio_files", "./results")
    
    return results


# 自定义处理示例
def custom_process_example():
    """自定义处理示例"""
    processor = BatchAudioProcessor("base")  # 使用base模型
    
    audio_files = processor.get_audio_files("./audio_files")
    
    for audio_file in audio_files:
        result = processor.process_single_audio(audio_file, segment_length=60)  # 60秒片段
        
        if result['status'] == 'success':
            print(f"{result['file_name']}: {result['detected_language']} - {result['text'][:50]}...")
        else:
            print(f"处理失败: {result['file_name']}")
    
    return True