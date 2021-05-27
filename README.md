# segmetation_dicom_python

Модуль для сегментации.



## Файловая структура датасета


```
data_3
└── N_RLAD...
    └── gt
        └── meshmixer_proj.mix
            N_RLAD...body_Segment_1.nii - аннотация для тела и трубки
            N_RLAD...body_Segment_1.stl
            stick_transformed.stl
        pred
        └── Lung segmentation.nii - предсказание модуля слайсера
            Pred_segmentation.nii - наше предсказание 
        N_RLAD...   - папка со исходным сканом
        synt_data   - папка с синтетическими данными
        N_RLAD.._body.nii.gz - аннотация только тела

    ...
```
