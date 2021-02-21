#pragma once

#include <QMainWindow>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include "typedefs.hpp"

namespace QtCharts
{
class QChartView;
class QChart;
class QScatterSeries;
}

class QListWidget;

class InteractiveTrackViewer : public QMainWindow
{
    Q_OBJECT

    const std::size_t m_num_background_data = 10000ul; // 100'000ul;

    std::mt19937 m_rnd_gen;

    std::vector<RowTuple> m_data;
    std::vector<std::pair<float, std::span<RowTuple>>> m_tracks;

    QtCharts::QChartView *m_chart_view;
    QtCharts::QChart *m_chart;
    std::vector<QtCharts::QScatterSeries *> m_current_track;
    QListWidget *m_list_widget;
    

  public:
    explicit InteractiveTrackViewer(QWidget *parent = nullptr);
    ~InteractiveTrackViewer();

  private slots:
    void open();
    void exit();
};
