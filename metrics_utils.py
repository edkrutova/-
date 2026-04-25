# metrics_utils.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import time

class MetricsCollector:
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true, y_pred, y_score=None, prefix=''):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'predicted_anomalies': int(np.count_nonzero(y_pred)),
            'true_anomalies': int(np.count_nonzero(y_true)),
        }
        
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro' , zero_division=0)
        metrics['precision_macro'] = float(prec)
        metrics['recall_macro'] = float(rec)
        metrics['f1_macro'] = float(f1)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        auc = np.nan
        if y_score is not None:
            try:
                y_score = np.asarray(y_score).ravel()
                if len(np.unique(y_true)) >= 2:
                    auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                auc = np.nan
        metrics['roc_auc_macro'] = auc

        
        if prefix:
            self.results[prefix] = metrics
        return metrics
    
    def print_metrics(self, metrics, title=''):
        if title:
            print(f"\n{'='*60}\n{title:^60}\n{'='*60}")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro):  {metrics['f1_macro']:.4f}")
        roc = metrics.get('roc_auc_macro', np.nan)
        if roc is None or (isinstance(roc, float) and np.isnan(roc)):
            print("ROC-AUC  (macro):  NaN (no scores provided)")
        else:
            print(f"ROC-AUC  (macro):  {roc:.4f}")
        print(f"Predicted anomalies: {metrics['predicted_anomalies']}")
        print(f"True anomalies:      {metrics['true_anomalies']}")

class Timer:
    def __init__(self):
        self.times = {}
    
    def start(self, name):
        self.times[name] = {'start': time.time()}
    
    def stop(self, name):
        if name in self.times:
            self.times[name]['end'] = time.time()
            self.times[name]['duration'] = self.times[name]['end'] - self.times[name]['start']
    
    def get_duration(self, name):
        return self.times.get(name, {}).get('duration', 0)
    
    def print_times(self):
        print(f"\n{'='*60}\n{'TIMING':^60}\n{'='*60}")
        for name, info in self.times.items():
            if 'duration' in info:
                print(f"{name:30s}: {info['duration']:8.2f} sec")
