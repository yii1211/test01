import mne,os,re
import numpy as np
# import pywt
from scipy.signal import hilbert, butter, filtfilt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# --------------------------
# 原始数据读取模块（用户提供）
# --------------------------
def read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())



def readbdfdata(filename, pathname):
    '''
    Parameters
    ----------

    filename: list of str

    pathname: list of str

    Return:
    ----------
    eeg dictionary

    '''

    eeg = dict(data = [], events=[], srate = [],ch_names = [],nchan=[])

    if 'edf' in filename[0]:  ## DSI
        raw = mne.io.read_raw_edf(os.path.join(pathname[0],filename[0]))
        data, _ = raw[:-1]
        events = mne.find_events(raw)
        ch_names = raw.info['ch_names']
        fs = raw.info['sfreq']
        nchan = raw.info['nchan']
    else:    ## Neuracle
        ## read data
        raw = mne.io.read_raw_bdf(os.path.join(pathname[0],'data.bdf'), preload=False)
        ch_names = raw.info['ch_names']
        data, _ = raw[:len(ch_names)]
        fs = raw.info['sfreq']
        nchan = raw.info['nchan']
        ## read events
        try:
            annotationData = mne.io.read_raw_bdf(os.path.join(pathname[0],'evt.bdf'))
            try:
                tal_data = annotationData._read_segment_file([],[],0,0,int(annotationData.n_times),None,None)
                print('mne version <= 0.20')
            except:
                idx = np.empty(0, int)
                tal_data = annotationData._read_segment_file(np.empty((0, annotationData.n_times)), idx, 0, 0, int(annotationData.n_times), np.ones((len(idx), 1)), None)
                print('mne version > 0.20')
            onset, duration, description = read_annotations_bdf(tal_data[0])
            onset = np.array([i*fs for i in onset], dtype=np.int64)
            duration = np.array([int(i) for i in duration], dtype=np.int64)
            desc = np.array([int(i) for i in description], dtype= np.int64)
            events = np.vstack((onset,duration,desc)).T
        except:
            print('not found any event')
            events = []

    eeg['data'] = data
    eeg['events'] = events
    eeg['srate'] = fs
    eeg['ch_names'] = ch_names
    eeg['nchan'] = nchan
    return eeg

# --------------------------
# 预处理模块
# --------------------------
class EEGPreprocessor:
    def __init__(self):
        self.raw = None

    def process(self, raw_eeg):
        """EEG预处理流程"""
        try:
            # 移除T7/T8通道
            raw_eeg.drop_channels(['T7', 'T8'])


            # 陷波滤波（50Hz）
            raw_eeg.notch_filter(50, picks='eeg')

            # 带通滤波（0.5-30Hz）
            raw_eeg.filter(0.5, 30, picks='eeg')

            # 平均参考
            raw_eeg.set_eeg_reference(ref_channels='average')

            # 分段为40秒Epoch
            events = mne.make_fixed_length_events(raw_eeg, duration=40)
            epochs = mne.Epochs(raw_eeg, events, tmin=0, tmax=40, baseline=None, preload=True)

            # 剔除异常Epoch
            epochs.drop_bad(reject={'eeg': 100e-6})

            # 选择前4个干净Epoch
            return epochs[:4]
        except Exception as e:
            print(f"EEG预处理失败: {str(e)}")
            return None


class fNIRSPreprocessor:
    def process(self, hbo_signal, fs=10):
        """fNIRS预处理流程"""
        try:
            # 带通滤波
            b, a = butter(2, [0.01, 0.1], btype='band', fs=fs)
            filtered = filtfilt(b, a, hbo_signal)

            # 多项式去漂移
            coeffs = np.polyfit(np.arange(len(filtered)), filtered, 3)
            trend = np.polyval(coeffs, np.arange(len(filtered)))
            detrended = filtered - trend

            # 小波去噪
            coeffs = pywt.wavedec(detrended, 'db4', level=5)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(detrended)))
            coeffs = [pywt.threshold(c, threshold, 'soft') for c in coeffs]
            denoised = pywt.waverec(coeffs, 'db4')

            # 截取稳定数据
            return denoised[30 * fs: 30 * fs + 180 * fs]
        except Exception as e:
            print(f"fNIRS预处理失败: {str(e)}")
            return None


# --------------------------
# 特征提取模块
# --------------------------
class FeatureExtractor:
    def __init__(self):
        self.eeg_features = []
        self.fnirs_features = []

    def extract_eeg_features(self, epochs):
        """提取EEG特征"""
        # 计算PLV矩阵
        plv_matrix = self._calculate_plv(epochs)

        # 图论特征
        graph_features = self._graph_analysis(plv_matrix)

        # 功率不对称特征
        asymm_features = self._calculate_asymmetry(epochs)

        return np.concatenate([graph_features, asymm_features])

    def extract_fnirs_features(self, hbo_signal):
        """提取fNIRS特征"""
        # 样本熵
        sampen = sample_entropy(hbo_signal, order=2)

        # 功能连接（示例）
        corr_matrix = np.corrcoef(hbo_signal.T)

        return np.concatenate([sampen, corr_matrix.flatten()])

    def _calculate_plv(self, epochs):

    # ...（实现PLV计算）...

    def _graph_analysis(self, plv_matrix):

    # ...（实现图论特征计算）...

    def _calculate_asymmetry(self, epochs):


# ...（实现功率不对称计算）...

# --------------------------
# 主处理流程
# --------------------------
def full_pipeline(root_dir, output_dir):
    # 初始化模块
    eeg_processor = EEGPreprocessor()
    fnirs_processor = fNIRSPreprocessor()
    extractor = FeatureExtractor()

    all_features = []
    labels = []
    error_log = []

    # 遍历所有被试
    subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for subj in tqdm(subjects, desc="Processing Subjects"):
        subj_dir = os.path.join(root_dir, subj)
        try:
            # 读取原始数据
            raw_data = readbdfdata(["data.bdf"], [subj_dir])

            # EEG处理
            clean_epochs = eeg_processor.process(raw_data['data'])
            if clean_epochs is None:
                raise ValueError("无效的EEG数据")

            # fNIRS处理
            hbo = ...  # 从原始数据获取fNIRS信号
            processed_fnirs = fnirs_processor.process(hbo)
            if processed_fnirs is None:
                raise ValueError("无效的fNIRS数据")

            # 特征提取
            eeg_feat = extractor.extract_eeg_features(clean_epochs)
            fnirs_feat = extractor.extract_fnirs_features(processed_fnirs)
            combined_feat = np.concatenate([eeg_feat, fnirs_feat])

            # 存储特征和标签
            all_features.append(combined_feat)
            labels.append(0 if "Depression" in subj_dir else 1)  # 假设路径包含标签信息

        except Exception as e:
            error_log.append({
                "subject": subj,
                "error": str(e)
            })
            continue

    # 特征选择与分类
    X = np.array(all_features)
    y = np.array(labels)

    # LASSO特征选择
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    selected = np.where(lasso.coef_ != 0)[0]

    # SVM分类
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1))
    ])

    # LOOCV验证
    loo = LeaveOneOut()
    accuracies = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train[:, selected], y_train)
        accuracies.append(model.score(X_test[:, selected], y_test))

    # 保存结果
    results = {
        "accuracy": np.mean(accuracies),
        "features": X,
        "labels": y,
        "selected_features": selected,
        "errors": error_log
    }
    np.savez(os.path.join(output_dir, "results.npz"), **results)

    print(f"\n最终分类准确率: {np.mean(accuracies) * 100:.2f}%")
    print(f"错误日志保存至: {os.path.join(output_dir, 'error_log.txt')}")
    with open(os.path.join(output_dir, 'error_log.txt'), 'w') as f:
        for err in error_log:
            f.write(f"{err['subject']}: {err['error']}\n")


if __name__ == "__main__":
    # 配置参数
    DATA_ROOT = r"D:\yili\history\Depression\Data_health\EEG\rawdata"
    OUTPUT_DIR = r"./processed_results"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 执行全流程
    # full_pipeline(DATA_ROOT, OUTPUT_DIR)