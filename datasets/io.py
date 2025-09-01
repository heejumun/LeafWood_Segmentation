import h5py
import numpy as np
import open3d
import os
import pandas as pd
import json

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.csv']:
            return cls._read_csv(file_path)
        elif file_extension in ['.json']:
            return cls._read_json(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud
    
    @classmethod
    def _read_csv(cls, file_path):
        # 데이터 읽기
        df = pd.read_csv(file_path, encoding='utf-8-sig', header=None)
        
        # # 첫 번째 행이 컬럼 이름(예: 'X', 'Y', 'Z')인지 확인
        # if df.iloc[0].str.match(r'^[A-Za-z]+$').all():  # 문자열로 된 컬럼이 있는지 확인
        #     df = df[1:].reset_index(drop=True)  # 첫 번째 행을 삭제하고 인덱스 재설정
        
        return df
    
    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_json(cls, file_path):
        with open(file_path, 'r') as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"Expected a list in JSON, got {type(raw)}")

        return np.array(raw, dtype=np.float32)

    
    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]