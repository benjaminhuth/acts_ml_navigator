#pragma once

#include <QListWidgetItem>
#include <span>
#include <tuple>
#include <vector>

#include <spdlog/spdlog.h>

#include "typedefs.hpp"

struct TrackListWidgetItem : public QListWidgetItem
{
    using TrackPtr = const std::pair<float, std::span<RowTuple>> *;

    const TrackPtr track = nullptr;

    TrackListWidgetItem(TrackPtr t, QListWidget *parent = nullptr) :
        QListWidgetItem(parent, ItemType::UserType),
        track(t)
    {}

    bool operator<(const QListWidgetItem &other) const override
    {
        auto other_ptr = dynamic_cast<const TrackListWidgetItem *>(&other);

        if( other_ptr )
        {
            return track->first < other_ptr->track->first;
        }
        else
        {
            return QListWidgetItem::operator<(other);
        }
    }
};
