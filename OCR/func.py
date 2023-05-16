import numpy as np

### 給定一個三維 array，內含4個點(x,y)。輸出1個array，內部有4個陣列分別為左上、右上、右下、左下點的座標。
def get_four_points(points:np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype='float32')
    sort_result = np.argsort(points.squeeze(), axis=0)
    # 先取左邊的點，接著在比y維度的值大小。便可知道誰是左上，誰是左下。
    if int(np.where(sort_result[:,1] == sort_result[0,0])[0]) > int(np.where(sort_result[:,1] == sort_result[1,0])[0]):
        rect[3] = points.squeeze()[sort_result[0,0]]
        rect[0] = points.squeeze()[sort_result[1,0]] 
    else:
        rect[3] = points.squeeze()[sort_result[1,0]]
        rect[0] = points.squeeze()[sort_result[0,0]]
    # 同理，取右邊的點。
    if int(np.where(sort_result[:,1] == sort_result[2,0])[0]) > int(np.where(sort_result[:,1] == sort_result[3,0])[0]):
        rect[2] = points.squeeze()[sort_result[2,0]]
        rect[1] = points.squeeze()[sort_result[3,0]] 
    else:
        rect[2] = points.squeeze()[sort_result[3,0]]
        rect[1] = points.squeeze()[sort_result[2,0]]
    return rect