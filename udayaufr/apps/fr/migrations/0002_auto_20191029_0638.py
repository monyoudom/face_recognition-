# Generated by Django 2.2.2 on 2019-10-29 06:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fr', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userfacedataset',
            name='age',
            field=models.CharField(max_length=2),
        ),
        migrations.AlterField(
            model_name='userfacedataset',
            name='gender',
            field=models.CharField(max_length=2),
        ),
        migrations.DeleteModel(
            name='Age',
        ),
        migrations.DeleteModel(
            name='Gender',
        ),
    ]
