from sklearn.pipeline import make_pipeline
from preprocessing import MakeCopyTransformer, LabelEncoderCol, DummiesVariable, SimpleImputerCol, DropColumnsTransformer

pipe_preprocessing = make_pipeline(MakeCopyTransformer(),
                                   LabelEncoderCol('Sex'),
                                   DummiesVariable('Embarked'),
                                   DummiesVariable('Pclass')
                                   )


pipe_fe = make_pipeline(MakeCopyTransformer(),
                        SimpleImputerCol('Age', 'mean'),
                        DropColumnsTransformer(['PassengerId', 'Pclass', 'Name', 'Ticket', 'Cabin', 'Embarked'])
                        )
