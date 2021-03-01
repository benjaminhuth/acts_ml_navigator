#pragma once

#include <QDialog>
#include <QPushButton>
#include <QWidget>
#include <QListWidget>
#include <QList>
#include <QListWidgetItem>
#include <QAbstractSeries>
#include <QPushButton>
#include <QVBoxLayout>


class SeriesSelectDialog final : public QDialog
{
    Q_OBJECT

    QListWidget* m_list_widget;
    QPushButton* m_back_button;
    
    QList<QtCharts::QAbstractSeries *> m_series;
    
public:
    SeriesSelectDialog(QList<QtCharts::QAbstractSeries *> series, QWidget *parent = nullptr);
};
