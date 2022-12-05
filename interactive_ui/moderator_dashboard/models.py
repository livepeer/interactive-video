from django.db import models

# Create your models here.
from datetime import datetime

from sqlalchemy import Table, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy.orm import relationship

STATEMENT_TEXT_MAX_LENGTH = 255


class ModelBase(object):
    """
    An augmented base class for SqlAlchemy models.
    """

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    created_at = Column(
        DateTime,
        default=datetime.now()
    )

    last_updated = Column(
        DateTime,
        default=datetime.now(),
        onupdate=datetime.now()
    )


Base = declarative_base(cls=ModelBase)


class SampleFaces(Base):
    """
    Sample face information model
    """
    __tablename__ = 'sample_faces'

    sample_id = Column(
        String(STATEMENT_TEXT_MAX_LENGTH),
        nullable=False
    )

    name = Column(
        String(STATEMENT_TEXT_MAX_LENGTH),
        nullable=False
    )

    meta_data = Column(
        Text(),
        nullable=True
    )

    action = Column(
        Text(),
        nullable=True
    )

    vectors = relationship('FeatureVectors', backref='face')
    imgdatas = relationship('FaceImages', backref='faceimg')


class FeatureVectors(Base):
    """
    Feature vectors of the registered sample faces
    """
    __tablename__ = 'feature_vectors'

    face_id = Column(
        Integer,
        ForeignKey('sample_faces.id', ondelete='CASCADE')
    )

    vector = Column(
        Text(),
        nullable=False
    )

class FaceImages(Base):

    #registered sample face images

    __tablename__ = 'face_images'

    face_id = Column(
        Integer,
        ForeignKey('sample_faces.id', ondelete='CASCADE')
    )

    imgdata = Column(
        BYTEA(),
        nullable=False
    )
