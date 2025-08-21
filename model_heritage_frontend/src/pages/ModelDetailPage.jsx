import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  FileText, 
  Users, 
  Clock, 
  Hash, 
  Database,
  TrendingUp,
  GitBranch,
  Info
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

export default function ModelDetailPage() {
  const { id } = useParams();
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModel();
  }, [id]);

  const fetchModel = async () => {
    try {
      const response = await fetch(`http://localhost:5001/api/models/${id}`);
      if (!response.ok) throw new Error('Model not found');
      const data = await response.json();
      setModel(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num?.toString() || '0';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ok': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <Link to="/models">
            <Button>Back to Models</Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <Link to="/models">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Models
          </Button>
        </Link>
        
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">{model.name}</h1>
            {model.description && (
              <p className="text-gray-600 max-w-2xl">{model.description}</p>
            )}
          </div>
          <Badge className={getStatusColor(model.status)}>
            {model.status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Info */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Specifications */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Info className="h-5 w-5" />
                <span>Model Specifications</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Total Parameters</label>
                  <p className="text-lg font-semibold">{formatNumber(model.total_parameters)}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Layer Count</label>
                  <p className="text-lg font-semibold">{model.layer_count || 'N/A'}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Structural Hash</label>
                  <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                    {model.structural_hash || 'N/A'}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Model ID</label>
                  <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                    {model.id}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Lineage Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <GitBranch className="h-5 w-5" />
                <span>Model Lineage</span>
              </CardTitle>
              <CardDescription>
                Parent-child relationships discovered through weight analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {model.lineage?.parent || model.lineage?.children?.length > 0 ? (
                <div className="space-y-4">
                  {/* Parent */}
                  {model.lineage.parent && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Parent Model</h4>
                      <Link to={`/models/${model.lineage.parent.id}`}>
                        <Card className="hover:shadow-md transition-shadow cursor-pointer">
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-medium">{model.lineage.parent.name}</p>
                                <p className="text-sm text-gray-500">
                                  {formatNumber(model.lineage.parent.total_parameters)} parameters
                                </p>
                              </div>
                              {model.confidence_score && (
                                <Badge variant="outline">
                                  {(model.confidence_score * 100).toFixed(1)}% confidence
                                </Badge>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </Link>
                    </div>
                  )}

                  {/* Children */}
                  {model.lineage.children?.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">
                        Child Models ({model.lineage.children.length})
                      </h4>
                      <div className="space-y-2">
                        {model.lineage.children.map((child) => (
                          <Link key={child.id} to={`/models/${child.id}`}>
                            <Card className="hover:shadow-md transition-shadow cursor-pointer">
                              <CardContent className="p-4">
                                <div className="flex items-center justify-between">
                                  <div>
                                    <p className="font-medium">{child.name}</p>
                                    <p className="text-sm text-gray-500">
                                      {formatNumber(child.total_parameters)} parameters
                                    </p>
                                  </div>
                                  {child.confidence_score && (
                                    <Badge variant="outline">
                                      {(child.confidence_score * 100).toFixed(1)}% confidence
                                    </Badge>
                                  )}
                                </div>
                              </CardContent>
                            </Card>
                          </Link>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <GitBranch className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>No lineage relationships discovered yet</p>
                  <p className="text-sm">This model appears to be standalone or analysis is still in progress</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Family Info */}
          {model.family_id && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="h-5 w-5" />
                  <span>Model Family</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Badge variant="outline" className="mb-3">
                  {model.family_id}
                </Badge>
                <p className="text-sm text-gray-600">
                  This model belongs to a family of structurally similar models.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Timestamps */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Timeline</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-500">Created</label>
                <p className="text-sm">
                  {new Date(model.created_at).toLocaleString()}
                </p>
              </div>
              {model.processed_at && (
                <div>
                  <label className="text-sm font-medium text-gray-500">Processed</label>
                  <p className="text-sm">
                    {new Date(model.processed_at).toLocaleString()}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Link to="/add-model" className="block">
                <Button variant="outline" className="w-full">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Add Related Model
                </Button>
              </Link>
              <Link to="/models" className="block">
                <Button variant="outline" className="w-full">
                  <Database className="h-4 w-4 mr-2" />
                  Browse All Models
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

